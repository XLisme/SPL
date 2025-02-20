from .discriminator import NLayerDiscriminator
from .generator import Generator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class GANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.generator = Generator(config).to(self.device)
        self.discriminator = NLayerDiscriminator(
            input_nc=3,
            ndf=64,
            n_layers=3
        ).to(self.device)
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config['learning_rate']
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config['learning_rate']
        )
        
    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            for poses, real_images in train_loader:
                poses = poses.to(self.device)
                real_images = real_images.to(self.device)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                fake_images = self.generator(poses)
                
                real_outputs = self.discriminator(real_images)
                real_loss = sum(self.adversarial_loss(out, torch.ones_like(out)) for out in real_outputs)
                
                fake_outputs = self.discriminator(fake_images.detach())
                fake_loss = sum(self.adversarial_loss(out, torch.zeros_like(out)) for out in fake_outputs)
                
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                fake_outputs = self.discriminator(fake_images)
                g_loss = sum(self.adversarial_loss(out, torch.ones_like(out)) for out in fake_outputs)
                g_loss.backward()
                self.g_optimizer.step()
                
            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch)
                
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
        }
        path = f"models/checkpoints/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, path)
