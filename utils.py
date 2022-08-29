import cv2
import yaml
import glob
import os.path as osp
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


def load_yaml(config_file):
    config_obj = {}
    if osp.exists(config_file):
        with open(config_file, 'r') as f:
                config_obj = yaml.safe_load(f)
    return config_obj


class CompCarsData(Dataset):
    def __init__(self, root, transforms_, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files = glob.glob(f'{root}/cars_test/*.jpg') + glob.glob(f'{root}/cars_train/*.jpg')

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        img = self.transform(Image.open(filepath).convert('RGB'))
        # filename = filepath.split('/')[-1]
        return img

    def __len__(self):
        return len(self.files)