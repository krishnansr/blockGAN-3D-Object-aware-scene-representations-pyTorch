from torch import nn

class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__()
        pass

    def forward(self, x):
        raise NotImplementedError

class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__()
        pass

    def forward(self, x):
        raise NotImplementedError
