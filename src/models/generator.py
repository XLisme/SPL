import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        
        # Số channels đầu vào (pose) và đầu ra (ảnh RGB)
        input_nc = 3  # Pose representation
        output_nc = 3  # RGB image
        ngf = config['ngf']  # Số filters cơ bản
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input layer
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            
            # Downsampling layers
            nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
            
            nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(True),
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(ngf*4) for _ in range(9)]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsampling layers
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            
            # Output layer
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
        # Attention module
        self.attention = SelfAttention(ngf*4)

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Apply residual blocks
        x = self.residual_blocks(x)
        
        # Decode
        x = self.decoder(x)
        
        return x

class ResidualBlock(nn.Module):
    """Residual Block với skip connection"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class SelfAttention(nn.Module):
    """Self Attention Module"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Query
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        
        # Key
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        
        # Attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        # Value
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)
        
        # Output
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        return self.gamma*out + x
