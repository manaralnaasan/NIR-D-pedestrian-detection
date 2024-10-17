import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1)
    
    def forward(self, x):
        avg_out = F.avg_pool2d(x, x.size(2))
        avg_out = F.relu(self.fc1(avg_out))
        avg_out = torch.sigmoid(self.fc2(avg_out))
        return x * avg_out

class PixelAttention(nn.Module):
    def __init__(self, in_planes):
        super(PixelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 1, kernel_size=1)

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv1(max_out))
        return x * scale

class PyramidPooling(nn.Module):
    def __init__(self, in_planes):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.conv1 = nn.Conv2d(in_planes, in_planes // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, in_planes // 4, kernel_size=1)

    def forward(self, x):
        size = x.size()[2:]
        pool1_out = F.interpolate(self.conv1(self.pool1(x)), size=size, mode='bilinear', align_corners=False)
        pool2_out = F.interpolate(self.conv2(self.pool2(x)), size=size, mode='bilinear', align_corners=False)
        return torch.cat([x, pool1_out, pool2_out], dim=1)
