# Unsupervised Discovery of Interpretable Directions in the GAN Latent Space

Authors official implementation of the [_Unsupervised Discovery of Interpretable Directions in the GAN Latent Space_](https://arxiv.org/abs/2002.03754) (ICML 2020).

This code explores interpretable latent space directions of a pretrained GAN.

![An image](./images/rect_icml2020_2.png)
_Our approach scheme: latent deformator A aims to produce shifts that are easy to distinguish for the reconstructor R_

Here are several examples for Spectal Norm GAN (MNIST & Anime Faces), ProgGAN (CelebA-HQ) and BigGAN (ILSVRC):
![An image](./images/tizer.png)

## Requirements
python 3.6 or later\
jupyter (for visualization)
>torch>=1.4\
torchvision\
tqdm\
tensorboardX

see `requirement.txt` for exact authors environment.

## Training

Here is a minimal example of latent rectification run command:
```
python run_train.py \
    --gan_type BigGAN \
    --gan_weights models/pretrained/generators/BigGAN/G_ema.pth \
    --deformator ortho \
    --out rectification_results_dir
```
this script will save the latent space directions stored in `LatentDeformator` module weights. \
It also saves images charts with latent directions examples.
`gan_type` specifies the generator model.

Note that you can pass as an argument any parameter of `Params` class defined in `trainer.py`

## Evaluation

Run `evaluation.ipynb` notebook for the discovered directions inspection.

## Pre-trained Models

Run `python download.py` to download all pretrained generators and latent directions.
We also add `human_annotation.txt` file with annotation of some of directions.

The pretrained models are the unchanged copies from the following source:
`100_celeb_hq_network-snapshot-010403.pth` from https://github.com/ptrblck/prog_gans_pytorch_inference
`G_ema.pth` from https://github.com/ajbrock/BigGAN-PyTorch and `stylegan2-ffhq-config-f.pkl` https://github.com/NVlabs/stylegan2
converted with https://github.com/rosinality/stylegan2-pytorch

## Results

Here are some examples of generated images manipulation by moving along discovered directions:

![An image](./images/stylegan2_kid2_eyes.gif)

_StyleGAN2 - FFHQ - opened eyes_

![An image](./images/bigbigan_mushroom_light.gif)

_BigBiGAN - ImageNet - light direction_

![An image](./images/bird_rotation.gif)

_BigGAN - ImageNet - rotation_

## Credits
BigGAN code and weights are based on the authors implementation:
https://github.com/ajbrock/BigGAN-PyTorch

ProgGAN code and weights are based on:
https://github.com/ptrblck/prog_gans_pytorch_inference

U-net segmentation model code is based on:
https://github.com/milesial/Pytorch-UNet
