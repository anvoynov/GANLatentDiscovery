import os
import json
import torch
from collections import OrderedDict

from run_train import DEFORMATOR_TYPE_DICT
from models.gan_load import make_big_gan, make_proggan, make_external
from latent_deformator import LatentDeformator
from latent_shift_predictor import ResNetShiftPredictor, LeNetShiftPredictor
from constants import WEIGHTS


def load_from_dir(root_dir, model_index=None, G_weights=None, verbose=False):
    args = json.load(open(os.path.join(root_dir, 'args.json')))

    models_dir = os.path.join(root_dir, 'models')
    if model_index is None:
        models = os.listdir(models_dir)
        model_index = max(
            [int(name.split('.')[0].split('_')[-1]) for name in models
             if name.startswith('deformator')])

        if verbose:
            print('using max index {}'.format(model_index))


    if G_weights is None:
        G_weights = args['gan_weights']
    if G_weights is None or not os.path.isfile(G_weights):
        if verbose:
            print('Using default local G weights')
        G_weights = WEIGHTS[args['gan_type']]

    if args['gan_type'] == 'BigGAN':
        G = make_big_gan(G_weights, args['target_class']).eval()
    elif args['gan_type'] in ['ProgGAN', 'PGGAN']:
        G = make_proggan(G_weights)
    else:
        G = make_external(G_weights)

    deformator = LatentDeformator(G.dim_z, type=DEFORMATOR_TYPE_DICT[args['deformator']])

    if 'shift_predictor' not in args.keys() or args['shift_predictor'] == 'ResNet':
        shift_predictor = ResNetShiftPredictor(G.dim_z)
    elif args['shift_predictor'] == 'LeNet':
        shift_predictor = LeNetShiftPredictor(G.dim_z, 1 if args['gan_type'] == 'SN_MNIST' else 3)

    deformator_model_path = os.path.join(models_dir, 'deformator_{}.pt'.format(model_index))
    shift_model_path = os.path.join(models_dir, 'shift_predictor_{}.pt'.format(model_index))
    if os.path.isfile(deformator_model_path):
        deformator.load_state_dict(torch.load(deformator_model_path))
    if os.path.isfile(shift_model_path):
        shift_predictor.load_state_dict(torch.load(shift_model_path))

    # try to load dims annotation
    directions_json = os.path.join(root_dir, 'directions.json')
    if os.path.isfile(directions_json):
        with open(directions_json, 'r') as f:
            directions_dict = json.load(f, object_pairs_hook=OrderedDict)
            setattr(deformator, 'directions_dict', directions_dict)


    return deformator.eval().cuda(), G.eval().cuda(), shift_predictor.eval().cuda()
