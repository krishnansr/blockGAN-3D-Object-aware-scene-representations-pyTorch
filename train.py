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

from utils import load_yaml, CompCarsDataset
from block_gan import Generator, Discriminator


DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'using torch device: {DEVICE}, torch ver: {torch.__version__}')


def train_model(config):
    # get hyperparams
    img_height = config.get('IMG_HEIGHT', 64)  # 128, 256
    lr = config.get('LR', 3e-4)  # 3e-4
    z_dim = config.get('Z_DIM', 128)  # 128, 256
    ngf = config.get('NGF', 64)
    ndf = config.get('NDF', 32)
    angles = config.get('ANGLES', [0,0,-45,45,0,0])
    batch_size = config.get('BATCH_SIZE', 32)
    num_epochs = config.get('EPOCHS', 50)
    seed = config.get('SEED', 123)
    num_workers = config.get('NUM_WORKERS', 1)
    image_dim = config.get('IMAGE_DIMS', (img_height, img_height, 1))
    image_dim = reduce(operator.mul, image_dim)
    torch.manual_seed(seed)

    disc = Discriminator(n_features=ndf, z_dim=z_dim).to(DEVICE)
    gen = Generator(n_features=ngf, z_dim=z_dim, angles=angles).to(DEVICE)
    fixed_noise = torch.randn((batch_size, z_dim)).to(DEVICE)
    data_transforms = [
        transforms.Resize(img_height),
        transforms.CenterCrop(img_height),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # dataset = datasets.MNIST(root="dataset/", transform=data_transforms, download=True)
    dataset = CompCarsDataset(root=r"C:/datasets/stanf_cars", transforms_=data_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr)  # use SGD?
    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()  # simulate minimax eq

    # tensorboard setup
    writer_fake = SummaryWriter(f"runs/GAN_CARS/fake")
    writer_real = SummaryWriter(f"runs/GAN_CARS/real")
    global_step = 0
    for epoch in range(num_epochs):
        for batch_idx, real in enumerate(tqdm(loader)):
            real = real.view(-1, image_dim).to(DEVICE)
            batch_size = real.shape[0]

            # train Discriminator: max log(D(real)) + log(1 - D(G(z)))
            # train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))

            # sample z
            # noise = torch.randn(batch_size, z_dim).to(DEVICE)  # z
            rnd_state = np.random.RandomState(seed)
            z = torch.from_numpy(
                rnd_state.normal(0, 1, size=(batch_size, z_dim))
            ).float()
            z = z.cuda()

            angles = gen.sample_angles(z.size(0), *angles)
            thetas = gen.get_theta(angles)
            if z.is_cuda:
                thetas = thetas.cuda()

            fake = gen(z, thetas)  # G(z)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx == 0:
                print(
                    f"\nEpoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise, thetas).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                    writer_fake.add_image(
                        "Cars Fake Images", img_grid_fake, global_step=global_step
                    )
                    writer_real.add_image(
                        "Cars Real Images", img_grid_real, global_step=global_step
                    )
                    writer_fake.add_scalar('Generator loss', lossG, global_step=global_step)
                    writer_real.add_scalar('Discriminator loss', lossD, global_step=global_step)
                    global_step += 1


if __name__ == "__main__":
    config_file = ''
    train_model(config=load_yaml(config_file))