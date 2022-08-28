# import warnings
# warnings.filterwarnings('always')

import cv2
import torch
import operator
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from functools import reduce
from progressbar import progressbar
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .utils import load_yaml
from .block_gan import Generator, Discriminator


DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'using torch device: {DEVICE}, torch ver: {torch.__version__}')

def train_model(config):
    # get hyperparams
    lr = config.get('LR', 2e-3)
    z_dim = config.get('Z_DIM', 64)
    batch_size = config.get('BATCH_SIZE', 32)
    num_epochs = config.get('EPOCHS', 50)
    image_dim = config.get('IMAGE_DIMS', (28, 28, 1))
    image_dim = reduce(operator.mul, image_dim)

    disc = Discriminator(image_dim).to(DEVICE)
    gen = Generator(z_dim, image_dim).to(DEVICE)
    fixed_noise = torch.randn((batch_size, z_dim)).to(DEVICE)
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])



if __name__ == "__main__":
    config_file = ''
    train_model(config=load_yaml(config_file))