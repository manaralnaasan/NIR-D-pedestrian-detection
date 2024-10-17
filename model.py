import torch
import torch.nn as nn
from torchvision import models
from fusion import ChannelAttention, PixelAttention, PyramidPooling

class RAFFNet(nn.Module):
    def __init__(self):
        super(RAFFNet, self).__init__()
        self.backbone_nir = models.resnet50(pretrained=True)
        self.backbone_depth = models.resnet50(pretrained=True)
        
        self.channel_attention = ChannelAttention(2048)
        self.pixel_attention = PixelAttention(2048)
        self.pyramid_pooling = PyramidPooling(2048)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(2048 + 1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, nir_input, depth_input):
        nir_feat = self.backbone_nir(nir_input)
        depth_feat = self.backbone_depth(depth_input)
        fused_feat = torch.cat((nir_feat, depth_feat), dim=1)
        ca_feat = self.channel_attention(fused_feat)
        pa_feat = self.pixel_attention(ca_feat)
        pp_feat = self.pyramid_pooling(pa_feat)
        output = self.decoder(pp_feat)
        return output
