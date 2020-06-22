import os
import json
import torch

from constants import DEFORMATOR_TYPE_DICT, HUMAN_ANNOTATION_FILE
from models.gan_load import make_big_gan, make_proggan, make_style_gan2, make_sngan
from latent_deformator import LatentDeformator
from latent_shift_predictor import LatentShiftPredictor, LeNetShiftPredictor
from constants import WEIGHTS


def load_generator(args, G_weights, shift_in_w):
    gan_type = args['gan_type']
    if gan_type == 'BigGAN':
        G = make_big_gan(G_weights, args['target_class']).eval()
    elif gan_type in ['ProgGAN']:
        G = make_proggan(G_weights)
    elif 'StyleGAN2' in gan_type:
        G = make_style_gan2(args['resolution'], G_weights, shift_in_w)
    else:
        G = make_sngan(G_weights)

    return G


def load_from_dir(root_dir, model_index=None, G_weights=None, shift_in_w=True):
    args = json.load(open(os.path.join(root_dir, 'args.json')))

    models_dir = os.path.join(root_dir, 'models')
    if model_index is None:
        models = os.listdir(models_dir)
        model_index = max(
            [int(name.split('.')[0].split('_')[-1]) for name in models
             if name.startswith('deformator')])

    if G_weights is None:
        G_weights = args['gan_weights']
    if G_weights is None or not os.path.isfile(G_weights):
        print('Using default local G weights')
        G_weights = WEIGHTS[args['gan_type']]
        if isinstance(G_weights, dict):
            G_weights = G_weights[str(args['resolution'])]

    if 'resolution' not in args.keys():
        args['resolution'] = 128

    G = load_generator(args, G_weights, shift_in_w)
    deformator = LatentDeformator(
        shift_dim=G.dim_shift,
        input_dim=args['directions_count'] if 'directions_count' in args.keys() else None,
        out_dim=args['max_latent_dim'] if 'max_latent_dim' in args.keys() else None,
        type=DEFORMATOR_TYPE_DICT[args['deformator']])

    if 'shift_predictor' not in args.keys() or args['shift_predictor'] == 'ResNet':
        shift_predictor = LatentShiftPredictor(G.dim_shift)
    elif args['shift_predictor'] == 'LeNet':
        shift_predictor = LeNetShiftPredictor(
            G.dim_shift, 1 if args['gan_type'] == 'SN_MNIST' else 3)

    deformator_model_path = os.path.join(models_dir, 'deformator_{}.pt'.format(model_index))
    shift_model_path = os.path.join(models_dir, 'shift_predictor_{}.pt'.format(model_index))
    if os.path.isfile(deformator_model_path):
        deformator.load_state_dict(
            torch.load(deformator_model_path, map_location=torch.device('cpu')))
    if os.path.isfile(shift_model_path):
        shift_predictor.load_state_dict(
            torch.load(shift_model_path, map_location=torch.device('cpu')))

    setattr(deformator, 'annotation', load_human_annotation(os.path.join(root_dir,
        HUMAN_ANNOTATION_FILE)))

    return deformator.eval().cuda(), G.eval().cuda(), shift_predictor.eval().cuda()


def load_human_annotation(txt_file, verbose=False):
    annotation_dict = {}
    if os.path.isfile(txt_file):
        with open(txt_file) as source:
            for line in source.readlines():
                indx_str, annotation = line.split(': ')
                if len(annotation) > 0:
                    i = 0
                    annotation_unique = annotation
                    while annotation_unique in annotation_dict.keys():
                        i += 1
                        annotation_unique = f'{annotation} ({i})'
                    annotation_unique = annotation_unique.replace('\n', '').replace(' ', '_')
                    annotation_dict[annotation_unique] = int(indx_str)
        if verbose:
            print(f'loaded {len(annotation_dict)} annotated directions from {txt_file}')

    return annotation_dict
