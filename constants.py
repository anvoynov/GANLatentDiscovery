from latent_deformator import DeformatorType
from trainer import ShiftDistribution, DeformatorLoss


DEFORMATOR_TYPE_DICT = {
    'fc': DeformatorType.FC,
    'linear': DeformatorType.LINEAR,
    'id': DeformatorType.ID,
    'ortho': DeformatorType.ORTHO,
    'proj': DeformatorType.PROJECTIVE,
    'random': DeformatorType.RANDOM,
}

DEFORMATOR_LOSS_DICT = {
    'l2': DeformatorLoss.L2,
    'relative': DeformatorLoss.RELATIVE,
    'stat': DeformatorLoss.STAT,
    None: DeformatorLoss.NONE,
}

SHIFT_DISTRIDUTION_DICT = {
    'normal': ShiftDistribution.NORMAL,
    'uniform': ShiftDistribution.UNIFORM,
    None: None
}

WEIGHTS = {
    'BigGAN': 'models/pretrained/BigGAN/138k/G_ema.pth',
    'ProgGAN': 'models/pretrained/ProgGAN/100_celeb_hq_network-snapshot-010403.pth',
    'SN_MNIST': 'models/pretrained/GANs/SN_MNIST',
    'Anime_64': 'models/pretrained/GANs/SN_Anime',
}
