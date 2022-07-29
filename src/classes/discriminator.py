from torch import nn

from functions.create_gan_label import get_disc_block

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.block1 = get_disc_block(1, 16, (3,3), 2)
        self.block2 = get_disc_block(16, 32, (5,5), 2)
        self.block3 = get_disc_block(32, 64, (5,5), 2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64, out_features=1)
    
    def forward(self, images):
        x1 = self.block1(images)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.flatten(x3)
        x5 = self.linear(x4)
        return x5