import os
import argparse
import json
import torch
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use("Agg")

from utils import save_command_run_params
from constants import DEFORMATOR_TYPE_DICT, WEIGHTS
from segmentation.unet_model import UNet
from segmentation.gan_segmentation import MaskGenerator, MaskSynthesizing, it_mask_gen
from segmentation.data import SegmentationDataset
from segmentation.metrics import model_metrics
from models.gan_load import make_big_gan
from latent_deformator import LatentDeformator
from segmentation.metrics import MAE, IoU
from segmentation.inference import SegmentationInference, Threshold


mask_synthesizing_dict = {
    'diff': MaskSynthesizing.DIFF,
    'intensity': MaskSynthesizing.INTENSITY,
}


class SegmentationTrainParams(object):
    def __init__(self, **kwargs):
        self.rate = 0.005
        self.weight_decay = 0.0
        self.steps_per_rate_decay = 4000
        self.rate_decay = 0.2

        self.latent_shift_r = 26
        self.mask_thr = 0.99
        self.use_diff = False

        self.batch_size = 64
        self.shifted_img_prob = 0.0

        self.n_steps = int(1e+4)
        self.steps_per_log = 100
        self.steps_per_checkpoint_save = 1000
        self.test_steps = 10

        self.mask_size_low = 0.05
        self.mask_size_up = 0.5

        for key, val in kwargs.items():
            if val is not None:
                self.__dict__[key] = val

        if isinstance(self.synthezing, str):
            self.synthezing = mask_synthesizing_dict[self.synthezing]


def train_segmentation(G, deformator, model, params, background_dim, out_dir,
                       gen_devices, val_dirs=None):
    model.train()
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, 'tensorboard'))

    params.batch_size = params.batch_size // len(gen_devices)

    mask_generator = MaskGenerator(
        G, deformator, background_dim, params).cuda().eval()
    # form test batch
    num_test_steps = params.test_samples_count // params.batch_size
    test_samples = [mask_generator() for _ in range(num_test_steps)]
    test_samples = [[s[0].cpu(), s[1].cpu()] for s in test_samples]

    optimizer = torch.optim.Adam(model.parameters(), lr=params.rate,
                                 weight_decay=params.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, params.steps_per_rate_decay, params.rate_decay)
    criterion = torch.nn.CrossEntropyLoss()

    start_step = 0
    checkpoint = os.path.join(out_dir, 'checkpoint.pth')
    if os.path.isfile(checkpoint):
        start_step = load_checkpoint(model, optimizer, lr_scheduler, checkpoint)
        print(f'Starting from step {start_step} checkpoint')

    print('start loop', flush=True)
    for step, (img, ref) in enumerate(it_mask_gen(mask_generator, torch.cuda.current_device())):
        step += start_step
        model.zero_grad()
        prediction = model(img)
        loss = criterion(prediction, ref)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if step > 0 and step % params.steps_per_checkpoint_save == 0:
            print(f'Step {step}: saving checkpoint')
            save_checkpoint(model, optimizer, lr_scheduler, step, checkpoint)

        if step % 10 == 0:
            writer.add_scalar('train/loss', loss.item(), step)

        if step > 0 and step % params.steps_per_log == 0:
            with torch.no_grad():
                loss = 0.0
                for img, ref in test_samples:
                    prediction = model(img.cuda())
                    loss += criterion(prediction, ref.cuda()).item()
            loss = loss / num_test_steps
            print(f'{int(100.0 * step / params.n_steps)}% | step {step}: {loss}')
            writer.add_scalar('val/loss', loss, step)

        is_val_step = \
            (step > 0 and step % params.steps_per_validation == 0) or (step == params.n_steps)
        if is_val_step and val_dirs is not None:
            print(f'Step: {step} | evaluation')
            model.eval()
            mae_stat = evaluate(SegmentationInference(model, resize_to=128),
                                val_dirs[0], val_dirs[1], (MAE,))
            iou_stat = evaluate(Threshold(model, resize_to=128),
                                val_dirs[0], val_dirs[1], (IoU,))

            update_out_json({**mae_stat, **iou_stat}, os.path.join(out_dir, 'score.json'))
            model.train()
        if step == params.n_steps:
            break

    model.eval()
    torch.save(model.state_dict(), os.path.join(out_dir, 'segmentation.pth'))


def save_checkpoint(model, opt, scheduler, step, checkpoint):
    state_dict = {
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step}
    torch.save(state_dict, checkpoint)


def load_checkpoint(model, opt, scheduler, checkpoint):
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict['model'])
    opt.load_state_dict(state_dict['opt'])
    scheduler.load_state_dict(state_dict['scheduler'])
    return state_dict['step']


def update_out_json(eval_dict, out_json):
    out_dict = {}
    if os.path.isfile(out_json):
        with open(out_json, 'r') as f:
            out_dict = json.load(f)

    with open(out_json, 'w') as out:
        out_dict.update(eval_dict)
        json.dump(out_dict, out)


def evaluate(segmentation_model, images_dir, masks_dir, metrics, size=None):
    segmentation_dl = torch.utils.data.DataLoader(
        SegmentationDataset(images_dir, masks_dir, size=size, crop=False), 1, shuffle=False)

    eval_out = model_metrics(segmentation_model, segmentation_dl, stats=metrics)
    print('Segmenation model', eval_out)

    return eval_out


def keys_in_dict_tree(dict_tree, keys):
    for key in keys:
        if key not in dict_tree.keys():
            return False
        dict_tree = dict_tree[key]
    return True


def main():
    parser = argparse.ArgumentParser(description='GAN-based unsupervised segmentation train')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--gan_weights', type=str, default=WEIGHTS['BigGAN'])
    parser.add_argument('--deformator_weights', type=str, required=True)
    parser.add_argument('--deformator_type', type=str,
                        choices=DEFORMATOR_TYPE_DICT.keys(), required=True)
    parser.add_argument('--background_dim', type=int, required=True)
    parser.add_argument('--classes', type=int, nargs='*', default=list(range(1000)))
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)

    parser.add_argument('--val_images_dir', type=str, default=None)
    parser.add_argument('--val_masks_dir', type=str, default=None)

    for key, val in SegmentationTrainParams().__dict__.items():
        val_type = type(val) if key is not 'synthezing' else str
        parser.add_argument('--{}'.format(key), type=val_type, default=None)

    args = parser.parse_args()
    torch.random.manual_seed(args.seed)

    torch.cuda.set_device(args.device)
    # save run p
    save_command_run_params(args)

    if len(args.classes) == 0:
        print('using all ImageNet')
        args.classes = list(range(1000))
    G = make_big_gan(args.gan_weights, args.classes).eval().cuda()
    deformator = LatentDeformator(G.dim_z, type=DEFORMATOR_TYPE_DICT[args.deformator_type])
    deformator.load_state_dict(
        torch.load(args.deformator_weights, map_location=torch.device('cpu')))
    deformator.eval().cuda()

    model = UNet().train().cuda()
    train_params = SegmentationTrainParams(**args.__dict__)
    print(f'run train with p: {train_params.__dict__}')

    train_segmentation(
        G, deformator, model, train_params, args.background_dim, args.out,
        val_dirs=[args.val_images_dirs, args.val_masks_dirs])


if __name__ == '__main__':
    main()
