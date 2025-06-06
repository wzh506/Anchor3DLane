import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import dtype
from mmseg.utils.utils import *
from mmseg.models.networks.feature_extractor import *
from mmseg.models.networks import Lane2D, Lane3D
from mmseg.models.networks.libs.layers import *
from mmseg.models.networks.PE import PositionEmbeddingLearned
from mmseg.models.networks.Layers import EncoderLayer
from mmseg.models.networks.Unet_parts import Down, Up
from .AAConv import AAConv
from ..builder import NECKS

# from AAConv import AAConv
# from ..builder import NECKS


class AAConv_Block(nn.Module):
    def __init__(self, in_planes, flag=False):
        super(AAConv_Block, self).__init__()
        self.flag = flag
        self.conv1 = AAConv(in_planes, in_planes, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = AAConv(in_planes, in_planes, 3, 1, 1)
 
    def forward(self, x, epoch, hw_range):
        res = self.conv1(x, epoch, hw_range)
        res = self.relu(res)
        res = self.conv2(res, epoch, hw_range)
        x = x + res
        return x


# 注意：我们这里输入是一个list
# 有个问题：是否需要在backbone之后再次进行一次卷积操作
#(1,256,45,60)
#(1,128,45,60)
#(1,64,90,120)
@NECKS.register_module()
class ADNet(nn.Module):
    def __init__(self, channels,output_channels=1,hw_range=[0,18]): #channels=256
        super(ADNet, self).__init__()
        # self.head_conv = nn.Conv2d(channels, 32, 3, 1, 1)
        # self.rb1 = AAConv_Block(32, flag=True)
        # self.down1 = ConvDown(32)
        # self.rb2 = AAConv_Block(64)
        # self.down2 = ConvDown(64)
        self.rb3 = AAConv_Block(channels)
        self.up1 = ConvUp(channels//2)
        self.fuse1 = Fusion(channels//2, channels)
        self.rb4 = AAConv_Block(channels//2)
        self.up2 = ConvUp(channels//2)
        self.rb5 = AAConv_Block(channels//4)
        self.tail_conv = nn.Conv2d(channels//4, output_channels, 3, 1, 1)
        self.hw_range = hw_range
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, inputs, epoch=0):
        # x1 = torch.cat([pan, lms], dim=1)
        
        # x1 = self.head_conv(x1)
        # x1 = self.rb1(x1, epoch, hw_range)

        # x2 = self.down1(x1)
        # x2 = self.rb2(x2, epoch, hw_range)

        # x3 = self.down2(x2)
        # x3 = self.rb3(x3, epoch, hw_range)
        # 已经通过Resnet的生成好了
        assert len(inputs) >= 3, "Expected at least 3 inputs, got {}".format(len(inputs))
        x1 = inputs[0]
        x2 = inputs[1]
        x3 = inputs[2]


        # x4 = self.up1(x3, x2) 
        x4 = self.rb3(x3, epoch, self.hw_range)
        x4 = self.fuse1(x2, x4) #256和128

        del x2
        x4 = self.rb4(x4, epoch, self.hw_range)
        x5 = self.up2(x4, x1)
        del x1
        x5 = self.rb5(x5, epoch, self.hw_range)
        x5 = self.tail_conv(x5)
        x5 = self.sigmoid(x5)
        return x5

# Downsample and Upsample blocks for Unet-strucutre

class ConvDown(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
 
        if dsconv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, in_channels * 2, 1, 1, 0)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 2, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels * 2, 3, 1, 1)
            )
 
    def forward(self, x):
        return self.conv(x)
    
class Fusion(nn.Module):
    def __init__(self, low_ch=128, high_ch=256, dsconv=True):
        super().__init__()
        
        # 通道对齐模块
        self.channel_align = nn.Sequential(
            nn.Conv2d(high_ch, low_ch, 1),  # 1x1卷积降维
            nn.BatchNorm2d(low_ch),
            nn.LeakyReLU(0.2)
        )
        
        # 特征融合模块
        self.fuse_conv = nn.Sequential(
            DepthwiseSeparableConv(low_ch, low_ch) if dsconv else nn.Conv2d(low_ch, low_ch, 3, padding=1),
            nn.BatchNorm2d(low_ch)
        )

    def forward(self, low_feat, high_feat):
        """
        Inputs:
            low_feat:  [B, 128, H, W] 低频特征
            high_feat: [B, 256, H, W] 高频特征
        Output:
            fused:     [B, 128, H, W] 融合结果
        """
        # 通道对齐
        high_feat = self.channel_align(high_feat)
        
        # 特征融合
        fused = low_feat + high_feat  # 残差连接
        return F.leaky_relu(self.fuse_conv(fused), 0.2)

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积，参数量减少75%"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, 
                                  padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
 
 
 
class ConvUp(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
 
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2, 0)
        if dsconv:
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2, bias=False),
                nn.Conv2d(in_channels // 2, in_channels // 2, 1, 1, 0)
            )
        else:
            self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1)
 
    def forward(self, x, y):
        x = F.leaky_relu(self.conv1(x))
        x = x + y
        x = F.leaky_relu(self.conv2(x))
        return x
    


if __name__ == '__main__':  
    # Test the AAConv_Block
    # input_tensor = torch.randn(1, 256, 45, 64)  # Example input tensor
    inputs = [torch.randn(1, 64, 90, 120),torch.randn(1, 128, 45, 60),torch.randn(1, 256, 45, 60)] #和openlane的输入保持一致 
    epoch = 0
    hw_range = [0,18]  # Replace with actual hw_range if needed

    model = ADNet(channels=256, output_channels=1)
    output_tensor = model(inputs,epoch, hw_range)
    print(output_tensor.shape)  # Check the output shape
    