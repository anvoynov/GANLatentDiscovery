import os
import sys
import argparse
import json
import random
import torch

import matplotlib
matplotlib.use("Agg")

from constants import DEFORMATOR_TYPE_DICT, DEFORMATOR_LOSS_DICT, SHIFT_DISTRIDUTION_DICT, WEIGHTS
from models.gan_load import make_big_gan, make_proggan, make_external
from latent_deformator import LatentDeformator
from latent_shift_predictor import ResNetShiftPredictor, LeNetShiftPredictor
from trainer import Trainer, Params


def main():
    parser = argparse.ArgumentParser(description='Latent space rectification')
    for key, val in Params().__dict__.items():
        parser.add_argument('--{}'.format(key), type=type(val), default=None)

    parser.add_argument('--args', type=str, default=None, help='json with all arguments')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--gan_type', type=str, choices=WEIGHTS.keys())
    parser.add_argument('--gan_weights', type=str, default=None)
    parser.add_argument('--target_class', type=int, default=239)
    parser.add_argument('--json', type=str)

    parser.add_argument('--deformator', type=str, default='ortho',
                        choices=DEFORMATOR_TYPE_DICT.keys())
    parser.add_argument('--deformator_random_init', type=bool, default=False)

    parser.add_argument('--shift_predictor_size', type=int)
    parser.add_argument('--shift_predictor', type=str,
                        choices=['ResNet', 'LeNet'], default='ResNet')
    parser.add_argument('--shift_distribution_key', type=str,
                        choices=SHIFT_DISTRIDUTION_DICT.keys())

    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)

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

    # init models
    if args.gan_weights is not None:
        weights_path = args.gan_weights
    else:
        weights_path = WEIGHTS[args.gan_type]

    if args.gan_type == 'BigGAN':
        G = make_big_gan(weights_path, args.target_class).eval()
    elif args.gan_type == 'ProgGAN':
        G = make_proggan(weights_path).eval()
    else:
        G = make_external(weights_path).eval()

    deformator = LatentDeformator(G.dim_z,
                                  type=DEFORMATOR_TYPE_DICT[args.deformator],
                                  random_init=args.deformator_random_init).cuda()

    if args.shift_predictor == 'ResNet':
        shift_predictor = ResNetShiftPredictor(G.dim_z, args.shift_predictor_size).cuda()
    elif args.shift_predictor == 'LeNet':
        shift_predictor = LeNetShiftPredictor(
            G.dim_z, 1 if args.gan_type == 'SN_MNIST' else 3).cuda()

    # training
    args.shift_distribution = SHIFT_DISTRIDUTION_DICT[args.shift_distribution_key]
    args.deformation_loss = DEFORMATOR_LOSS_DICT[args.deformation_loss]
    trainer = Trainer(params=Params(**args.__dict__), out_dir=args.out, out_json=args.json)
    trainer.train(G, deformator, shift_predictor)


if __name__ == '__main__':
    main()
