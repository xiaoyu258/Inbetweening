import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvOffset2D(nn.Conv2d):
    def __init__(self, channels, **kwargs):
        super(ConvOffset2D, self).__init__(channels, channels * 2, kernel_size=3, padding=1, bias=False, **kwargs)

    def forward(self, x):
        
        device = x.device
        B, C, H, W = x.size()
        
        offsets = super(ConvOffset2D, self).forward(x)
        
        # x: (b*c, h, w)  offsets: (b*c, h, w, 2)
        x = x.contiguous().view(-1, H, W)
        offsets = offsets.contiguous().view(-1, H, W, 2)
        
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        gridX = torch.tensor(gridX, requires_grad=False).unsqueeze(0).expand_as(x).float().to(device)
        gridY = torch.tensor(gridY, requires_grad=False).unsqueeze(0).expand_as(x).float().to(device)
        
        # range -1 to 1
        gridX = 2 * (gridX/(W - 1.0) - 0.5) + offsets[:,:,:,0]
        gridY = 2 * (gridY/(H - 1.0) - 0.5) + offsets[:,:,:,1]
        # stacking X and Y
        grid = torch.stack((gridX, gridY), dim=3)
        resampledx = torch.nn.functional.grid_sample(x.unsqueeze(1), grid, padding_mode="border", align_corners=True)        
        resampledx = resampledx.contiguous().view(-1, C, H, W)

        return resampledx



class down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize):
        
        super(down, self).__init__()
        # Initialize convolutional layers.
        self.offset1 = ConvOffset2D(inChannels)
        self.conv1 = nn.Conv2d(inChannels,  outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.offset2 = ConvOffset2D(outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
           
    def forward(self, x):
        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool2d(x, 2)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(self.offset1(x)), negative_slope = 0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(self.offset2(x)), negative_slope = 0.1)
        return x
    
    
class up(nn.Module):
    def __init__(self, inChannels, outChannels):
        
        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, 3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)
           
    def forward(self, x, skpCn):

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # Convolution + Leaky ReLU on (`x`, `skpCn`)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope = 0.1)
        return x



class DeformUNet(nn.Module):
    def __init__(self, inChannels, outChannels):
        
        super(DeformUNet, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1   = up(512, 512)
        self.up2   = up(512, 256)
        self.up3   = up(256, 128)
        self.up4   = up(128, 64)
        self.up5   = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)
        
    def forward(self, x):

        x  = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x  = self.down5(s5)
        x  = self.up1(x, s5)
        x  = self.up2(x, s4)
        x  = self.up3(x, s3)
        x  = self.up4(x, s2)
        x  = self.up5(x, s1)
        x  = F.leaky_relu(self.conv3(x), negative_slope = 0.1)
        return x