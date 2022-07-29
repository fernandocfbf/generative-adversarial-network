from torch import nn
import torch

from functions.compute_loss import fake_loss, real_loss
from functions.norm_weights import weights_init
from functions.create_gan_label import get_disc_block

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.block1 = get_disc_block(1, 16, (3,3), 2)
        self.block2 = get_disc_block(16, 32, (5,5), 2)
        self.block3 = get_disc_block(32, 64, (5,5), 2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64, out_features=1)

        self.optimizer = None
        self.total_loss = 0
    
    def get_optimizer(self, lr, beta_1, beta_2):
        discriminator_normalized = self.apply(weights_init)
        self.optimizer = torch.optim.Adam(
            discriminator_normalized.parameters(),
            lr = lr,
            betas = (beta_1, beta_2)
        )

    def forward(self, images):
        x1 = self.block1(images)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.flatten(x3)
        x5 = self.linear(x4)
        return x5
    
    def update(self, generator, batch_size, noise_dim, real_img):
        self.optimizer.zero_grad()
        noise = torch.randn(batch_size, noise_dim, device="cuda")

        fake_img = generator(noise)
        pred = self(fake_img)
        disc_fake_loss = fake_loss(pred)

        pred = self(real_img)
        disc_real_loss = real_loss(pred)

        total_loss = (disc_real_loss + disc_fake_loss)/2
        self.total_loss += total_loss
        total_loss.backward()
        self.optimizer.step()


