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

from ..builder import NECKS

@NECKS.register_module()
class BEVHead(nn.Module):
    def __init__(self, batch_norm=True, channels=128):
        super(BEVHead, self).__init__()
        self.size_reduce_layer_1 = Lane3D.SingleTopViewPathway(channels)            # 128 to 128
        self.size_reduce_layer_2 = Lane3D.SingleTopViewPathway(channels*2)          # 256 to 256
        self.size_dim_reduce_layer_3 = Lane3D.EasyDown2TopViewPathway(channels*4)   # 512 to 256

        self.dim_reduce_layers = nn.ModuleList()
        self.dim_reduce_layers.append(nn.Sequential(*make_one_layer(channels*2,     # 256
                                                        channels,                   # 128
                                                        kernel_size=1,
                                                        padding=0,
                                                        batch_norm=batch_norm)))
        self.dim_reduce_layers.append(nn.Sequential(*make_one_layer(channels*4,     # 512
                                                        channels*2,                 # 256
                                                        kernel_size=1,
                                                        padding=0,
                                                        batch_norm=batch_norm)))
        self.dim_reduce_layers.append(nn.Sequential(*make_one_layer(channels*4,     # 512
                                                        channels*2,                 # 256
                                                        kernel_size=1,
                                                        padding=0,
                                                        batch_norm=batch_norm)))

    def forward(self, projs):
        '''
            projs_0 size: torch.Size([4, 128, 208, 128])
            projs_1 size: torch.Size([4, 256, 104, 64])
            projs_2 size: torch.Size([4, 512, 52, 32])
            projs_3 size: torch.Size([4, 512, 26, 16])

            bev_feat_1 size: torch.Size([4, 128, 104, 64])
            bev_feat_2 size: torch.Size([4, 256, 52, 32])
            bev_feat_3 size: torch.Size([4, 256, 26, 16])

            bev_feat   size: torch.Size([4, 512, 26, 16])
        '''

        # bev_feat_1 = self.size_reduce_layer_1(projs[0])          # 128 -> 128
        # rts_proj_feat_1 = self.dim_reduce_layers[0](projs[1])    # 256 -> 128
        # bev_feat_2 = self.size_reduce_layer_2(torch.cat((bev_feat_1, rts_proj_feat_1), 1))     # 128+128 -> 256
        # rts_proj_feat_2 = self.dim_reduce_layers[1](projs[2])    # 512 -> 256
        # bev_feat_3 = self.size_dim_reduce_layer_3(torch.cat((bev_feat_2, rts_proj_feat_2), 1)) # 256+256 -> 256
        # rts_proj_feat_3 = self.dim_reduce_layers[2](projs[3])    # 512 -> 256
        # bev_feat = torch.cat((bev_feat_3, rts_proj_feat_3), 1)   # 256+256=512
        # return bev_feat
        bev_feat_1 = self.size_reduce_layer_1(projs[0])          # 128 -> 128
        return bev_feat_1