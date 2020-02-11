import os
import json
import torch
from GANs.models.sngan.generators.sn_gen_resnet import SN_RES_GEN_CONFIGS, make_resnet_generator
from GANs.models.sngan.generators.sn_sngen_resnet import make_snresnet_generator
from GANs.models.sngan.discriminators.sn_dis_resnet import SN_RES_DIS_CONFIGS, ResnetDiscriminator

from GANs.distributions import NormalDistribution


MODELS = {
    'fc': 28,
    'conv32': 32,
    'conv64': 64,
    'sn32': 32,
    'sn_fc': 28,
    'sn_resnet32': 32,
    'sn_resnet32_corrupted': 32,
    'sn_resnet48': 48,
    'sn_resnet64': 64,
    'sn_resnet128': 128,
    'sn_resnet256': 256,
}


DISTRIBUTIONS = {
    'normal': NormalDistribution,
}


class Args:
    def __init__(self, **kwargs):
        # old versions support
        self.nonfixed_noise = False
        self.noises_count = 1
        self.equal_split = False
        self.generator_batch_norm = False
        self.gen_sn = False
        self.distribution_params = "{}"

        self.__dict__.update(kwargs)


def make_sampler(distribution_name, dim, **kwargs):
    return DISTRIBUTIONS[distribution_name](dim, **kwargs)


def make_models(args):
    gen_kwargs = {
        'distribution': make_sampler(
            args.distribution,
            args.latent_dim,
            **dict(json.loads(args.distribution_params))
        ),
        'img_size': MODELS[args.model]
    }
    try:
        image_channels = args.image_channels
    except Exception:
        image_channels = 3

    if args.model.startswith('sn_snresnet'):
        gen_config = SN_RES_GEN_CONFIGS[args.model]
        generator = make_resnet_generator(gen_config, channels=image_channels, **gen_kwargs)

    elif args.model.startswith('sn_resnet'):
        gen_config = SN_RES_GEN_CONFIGS[args.model]
        if args.gen_sn:
            generator = make_snresnet_generator(gen_config, channels=image_channels, **gen_kwargs)
        else:
            generator = make_resnet_generator(gen_config, channels=image_channels, **gen_kwargs)

    return generator


def load_model_from_state_dict(root_dir, model_index=None, cuda=True):
    args = json.load(open(os.path.join(root_dir, 'args.json')))

    if model_index is None:
        models = os.listdir(root_dir)
        model_index = max(
            [int(name.split('.')[0].split('_')[-1]) for name in models
             if name.startswith('generator')])

        print('using generator generator_{}.pt'.format(model_index))
    generator_model_path = os.path.join(root_dir, 'generator_{}.pt'.format(model_index))

    args = Args(**args)
    generator, _ = make_models(args)
    generator.load_state_dict(
        torch.load(generator_model_path, map_location=torch.device('cpu')), strict=False)
    if cuda:
        generator.cuda()

    return generator
