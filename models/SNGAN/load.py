import os
import json
import torch
from models.SNGAN.sn_gen_resnet import SN_RES_GEN_CONFIGS, make_resnet_generator
from models.SNGAN.distribution import NormalDistribution


MODELS = {
    'sn_resnet32': 32,
    'sn_resnet64': 64,
}


DISTRIBUTIONS = {
    'normal': NormalDistribution,
}


class Args:
    def __init__(self, **kwargs):
        self.nonfixed_noise = False
        self.noises_count = 1
        self.equal_split = False
        self.generator_batch_norm = False
        self.gen_sn = False
        self.distribution_params = "{}"

        self.__dict__.update(kwargs)


def load_model_from_state_dict(root_dir):
    args = Args(**json.load(open(os.path.join(root_dir, 'args.json'))))
    generator_model_path = os.path.join(root_dir, 'generator.pt')

    try:
        image_channels = args.image_channels
    except Exception:
        image_channels = 3

    gen_config = SN_RES_GEN_CONFIGS[args.model]
    generator=  make_resnet_generator(gen_config, channels=image_channels,
                                      distribution=NormalDistribution(args.latent_dim),
                                      img_size=MODELS[args.model])

    generator.load_state_dict(
        torch.load(generator_model_path, map_location=torch.device('cpu')), strict=False)
    return generator
