import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        """
        Parameters:
            input_nc (int): Số channels của ảnh đầu vào (thường là 3 cho RGB)
            ndf (int): Số filters cơ bản của discriminator
            n_layers (int): Số lớp convolution
        """
        super(Discriminator, self).__init__()
        
        # Khởi tạo sequence model
        model = [
            # Layer đầu tiên
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        # Tăng số channels dần dần
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Layer cuối cùng để output probability map
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Output layer
        model += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()  # Để output nằm trong khoảng [0,1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
        Parameters:
            input (tensor): Batch của ảnh đầu vào, shape (batch_size, channels, height, width)
        Returns:
            tensor: Probability map cho biết từng vùng của ảnh là thật hay giả
        """
        return self.model(input)

class NLayerDiscriminator(nn.Module):
    """Discriminator với nhiều scales khác nhau (PatchGAN)"""
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        
        # Discriminator ở nhiều scales
        self.n_layers = n_layers
        self.discriminators = nn.ModuleList([
            Discriminator(input_nc, ndf, i+1) 
            for i in range(n_layers)
        ])
        
        # Weights cho mỗi scale
        self.weights = nn.Parameter(torch.ones(n_layers))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        Returns:
            list: Danh sách các probability maps ở các scales khác nhau
        """
        outputs = []
        for i, D in enumerate(self.discriminators):
            weight = self.sigmoid(self.weights[i])
            output = D(input) * weight
            outputs.append(output)
            
            # Downsample input cho scale tiếp theo
            if i < self.n_layers - 1:
                input = nn.functional.avg_pool2d(input, kernel_size=2, stride=2)
                
        return outputs
