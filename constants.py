from latent_deformator import DeformatorType
from trainer import ShiftDistribution


HUMAN_ANNOTATION_FILE = 'human_annotation.txt'


DEFORMATOR_TYPE_DICT = {
    'fc': DeformatorType.FC,
    'linear': DeformatorType.LINEAR,
    'id': DeformatorType.ID,
    'ortho': DeformatorType.ORTHO,
    'proj': DeformatorType.PROJECTIVE,
    'random': DeformatorType.RANDOM,
}


SHIFT_DISTRIDUTION_DICT = {
    'normal': ShiftDistribution.NORMAL,
    'uniform': ShiftDistribution.UNIFORM,
    None: None
}


WEIGHTS = {
    'BigGAN': 'models/pretrained/generators/BigGAN/G_ema.pth',
    'ProgGAN': 'models/pretrained/generators/ProgGAN/100_celeb_hq_network-snapshot-010403.pth',
    'SN_MNIST': 'models/pretrained/generators/SN_MNIST',
    'SN_Anime': 'models/pretrained/generators/SN_Anime',
    'StyleGAN2': 'models/pretrained/StyleGAN2/stylegan2-car-config-f.pt',
}
