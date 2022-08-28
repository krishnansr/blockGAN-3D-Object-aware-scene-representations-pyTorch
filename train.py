# import warnings
# warnings.filterwarnings('always')

import cv2
import torch
import operator
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

from tqdm import tqdm
from functools import reduce
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import load_yaml
from block_gan import Generator, Discriminator


DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'using torch device: {DEVICE}, torch ver: {torch.__version__}')


def train_model(config):
    # get hyperparams
    lr = config.get('LR', 2e-3)  # 3e-4
    z_dim = config.get('Z_DIM', 64)  # 128, 256
    batch_size = config.get('BATCH_SIZE', 32)
    num_epochs = config.get('EPOCHS', 50)
    image_dim = config.get('IMAGE_DIMS', (28, 28, 1))
    image_dim = reduce(operator.mul, image_dim)

    disc = Discriminator(image_dim).to(DEVICE)
    gen = Generator(z_dim, image_dim).to(DEVICE)
    fixed_noise = torch.randn((batch_size, z_dim)).to(DEVICE)
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # (0.5,), (0.5,)
    ])

    dataset = datasets.MNIST(root="dataset/", transform=data_transforms, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr)  # use SGD?
    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()  # simulate minimax eq

    # tensorboard setup
    writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
    writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
    global_step = 0
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(tqdm(loader)):
            real = real.view(-1, image_dim).to(DEVICE)
            batch_size = real.shape[0]

            # train Discriminator: max log(D(real)) + log(1 - D(G(z)))
            noise = torch.randn(batch_size, z_dim).to(DEVICE)  # z
            fake = gen(noise)  # G(z)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            # train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                    writer_fake.add_image(
                        "Mnist Fake Images", img_grid_fake, global_step=global_step
                    )
                    writer_real.add_image(
                        "Mnist Real Images", img_grid_real, global_step=global_step
                    )
                    writer_fake.add_scalar('Generator loss', lossG, global_step=global_step)
                    writer_real.add_scalar('Discriminator loss', lossD, global_step=global_step)
                    global_step += 1

if __name__ == "__main__":
    config_file = ''
    train_model(config=load_yaml(config_file))