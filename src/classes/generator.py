from torch import nn
import torch

from functions.create_gan_label import get_gen_block
from functions.norm_weights import weights_init
from functions.compute_loss import fake_loss, real_loss

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.block1 = get_gen_block(noise_dim, 256, (3,3), 2)
        self.block2 = get_gen_block(256, 128, (4,4), 1)
        self.block3 = get_gen_block(128, 64, (3,3), 2)
        self.block4 = get_gen_block(64, 1, (4,4), 2, final_block=True)

        self.optimizer = None
        self.total_loss = 0


    def get_optimizer(self, lr, beta_1, beta_2):
        generator_normalized = self.apply(weights_init)
        self.optimizer = torch.optim.Adam(
            generator_normalized.parameters(),
            lr = lr,
            betas = (beta_1, beta_2)
        )
    
    def forward(self, r_noise_vec):
        x = r_noise_vec.view(-1, self.noise_dim, 1, 1)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        return x4

    def update(self, discriminator, batch_size, noise_dim, real_img):
        self.optimizer.zero_grad()
        noise = torch.randn(batch_size, noise_dim, device="cuda")

        fake_img = self(noise)
        disc_pred = discriminator(fake_img)
        gen_real_loss = real_loss(disc_pred)

        total_loss = gen_real_loss
        self.total_loss += total_loss

        total_loss.backward()
        self.optimizer.step()

        return fake_img