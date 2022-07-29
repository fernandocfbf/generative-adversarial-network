from torch import nn

from functions.create_gan_label import get_gen_block

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.block1 = get_gen_block(noise_dim, 256, (3,3), 2)
        self.block2 = get_gen_block(256, 128, (4,4), 1)
        self.block3 = get_gen_block(128, 64, (3,3), 2)
        self.block4 = get_gen_block(64, 1, (4,4), 2, final_block=True)
    
    def forward(self, r_noise_vec):
        x = r_noise_vec.view(-1, self.noise_dim, 1, 1)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        return x4