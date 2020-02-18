import os
import sys
import argparse
import json
import torch
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use("Agg")

from constants import DEFORMATOR_TYPE_DICT, WEIGHTS

from segmentation.unet_model import UNet

from segmentation.gan_segmentation import MaskGenerator
from segmentation.data import SegmentationDataset
from segmentation.metrics import model_metrics
from models.gan_load import make_big_gan
from latent_deformator import LatentDeformator


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


def mix(x1, x2, p):
    assert x1.shape == x2.shape, 'x1.shape != x2.shape'
    mask = torch.rand([x1.shape[0]], device=x1.device) < p
    mix = torch.empty_like(x1)
    mix[mask] = x1[mask]
    mix[mask.logical_not()] = x2[mask.logical_not()]
    return mix


def train_segmentation(G, deformator, model, params, background_dim, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, 'tensorboard'))

    mask_generator = MaskGenerator(G, deformator, background_dim,
                                   params.latent_shift_r, params.mask_thr, params.use_diff)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.rate,
                                 weight_decay=params.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, params.steps_per_rate_decay, params.rate_decay)
    criterion = torch.nn.CrossEntropyLoss()

    start_step = 0
    checkpoint = os.path.join(out_dir, 'checkpoint.pth')
    if os.path.isfile(checkpoint):
        start_step = load_checkpoint(model, optimizer, lr_scheduler, checkpoint)
        print('Sartint from step {} checkpoint'.format(start_step))

    for step in range(start_step, params.n_steps, 1):
        z = torch.randn([params.batch_size, G.dim_z]).cuda()
        classes = G.mixed_classes(params.batch_size).cuda()

        with torch.no_grad():
            img = G(z, classes)
        img_neg, img_pos, ref = mask_generator(z=z, classes=classes)
        img = mix(img, img_pos, 1.0 - params.shifted_img_prob)

        prediction = model(img)

        if params.mask_size_low > 0.0 or params.mask_size_up < 1.0:
            ref_size = ref.shape[-2] * ref.shape[-1]
            ref_fraction = ref.sum(dim=[-1, -2]).to(torch.float) / ref_size
            mask = (ref_fraction > params.mask_size_low) & (ref_fraction < params.mask_size_up)
            if torch.sum(mask).item() < 2:
                continue
            prediction, ref = prediction[mask], ref[mask]

        loss = criterion(prediction, ref)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if step > 0 and step % params.steps_per_checkpoint_save == 0:
            print('Step {}: saving checkpoint'.format(step))
            save_checkpoint(model, optimizer, lr_scheduler, step, checkpoint)

        if step % 10 == 0:
            writer.add_scalar('train/loss', loss.item(), step)

        if step > 0 and step % params.steps_per_log == 0:
            with torch.no_grad():
                loss = 0.0
                for i in range(params.test_steps):
                    z = torch.randn([params.batch_size, G.dim_z]).cuda()
                    classes = G.mixed_classes(params.batch_size).cuda()
                    img = G(z, classes)
                    _, _, ref = mask_generator(z=z, classes=classes)

                    prediction = model(img)
                    loss += criterion(prediction, ref).item()

            loss = loss / params.test_steps
            print('{}% | step {}: {}'.format(
                int(100.0 * step / params.n_steps), step, loss))
            writer.add_scalar('val/loss', loss, step)

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


def evaluate(segmentation_model, images_dir, masks_dir, out_json, size=128):
    segmentation_model.eval()
    segmentation_dl = torch.utils.data.DataLoader(
        SegmentationDataset(images_dir, masks_dir, size=size, crop=False), 1, shuffle=False)

    iou, mae = model_metrics(segmentation_model, segmentation_dl).values()
    print('Segmenation model IoU: {:.3}, MAE: {:.3}'.format(iou, mae))
    with open(out_json, 'w') as out:
        json.dump({'IoU': iou, 'MAE': mae}, out)


def main():
    parser = argparse.ArgumentParser(description='GAN-based unsupervised segmentation train')
    parser.add_argument('--args', type=str, default=None, help='json with all arguments')

    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--gan_weights', type=str, default=WEIGHTS['BigGAN'])
    parser.add_argument('--deformator_weights', type=str, required=True)
    parser.add_argument('--deformator_type', type=str,
                        choices=DEFORMATOR_TYPE_DICT.keys(), required=True)
    parser.add_argument('--background_dim', type=int, required=True)
    parser.add_argument('--classes', type=int, nargs='*', default=list(range(1000)))
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)

    parser.add_argument('--val_images_dir', type=str)
    parser.add_argument('--val_masks_dir', type=str)

    for key, val in SegmentationTrainParams().__dict__.items():
        parser.add_argument('--{}'.format(key), type=type(val), default=None)

    args = parser.parse_args()
    torch.random.manual_seed(args.seed)

    torch.cuda.set_device(args.device)
    if args.args is not None:
        with open(args.args) as args_json:
            args_dict = json.load(args_json)
            args.__dict__.update(**args_dict)

    # save run params
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    with open(os.path.join(args.out, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    with open(os.path.join(args.out, 'command.sh'), 'w') as command_file:
        command_file.write(' '.join(sys.argv))
        command_file.write('\n')

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
    print('run train with params: {}'.format(train_params.__dict__))

    train_segmentation(G, deformator, model, train_params, args.background_dim, args.out)

    if args.val_images_dir is not None:
        evaluate(model, args.val_images_dir, args.val_masks_dir,
                 os.path.join(args.out, 'score.json'), 128)


if __name__ == '__main__':
    main()