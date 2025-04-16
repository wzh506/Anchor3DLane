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
class PerspectiveTransformer(nn.Module):
    def __init__(self, no_cuda, channels, bev_h, bev_w, uv_h, uv_w, M_inv, num_att, num_proj, nhead, npoints):
        super(PerspectiveTransformer, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.uv_h = uv_h
        self.uv_w = uv_w
        self.M_inv = M_inv
        self.num_att = num_att
        self.num_proj = num_proj
        self.nhead = nhead
        self.npoints = npoints

        self.query_embeds = nn.ModuleList()
        self.pe = nn.ModuleList()
        self.el = nn.ModuleList()
        self.project_layers = nn.ModuleList()
        self.ref_2d = []
        self.input_spatial_shapes = []
        self.input_level_start_index = []

        uv_feat_c = channels
        for i in range(self.num_proj):
            if i > 0:
                bev_h = bev_h // 2
                bev_w = bev_w // 2
                uv_h = uv_h // 2
                uv_w = uv_w // 2
                if i != self.num_proj-1:
                    uv_feat_c = uv_feat_c * 2

            bev_feat_len = bev_h * bev_w
            query_embed = nn.Embedding(bev_feat_len, uv_feat_c)
            self.query_embeds.append(query_embed)
            position_embed = PositionEmbeddingLearned(bev_h, bev_w, num_pos_feats=uv_feat_c//2)
            self.pe.append(position_embed)

            ref_point = self.get_reference_points(H=bev_h, W=bev_w, dim='2d', bs=1)
            self.ref_2d.append(ref_point)

            size_top = torch.Size([bev_h, bev_w])
            project_layer = Lane3D.RefPntsNoGradGenerator(size_top, self.M_inv, no_cuda)
            self.project_layers.append(project_layer)

            spatial_shape = torch.as_tensor([(uv_h, uv_w)], dtype=torch.long)
            self.input_spatial_shapes.append(spatial_shape)

            level_start_index = torch.as_tensor([0.0,], dtype=torch.long)
            self.input_level_start_index.append(level_start_index)

            for j in range(self.num_att):
                encoder_layers = EncoderLayer(d_model=uv_feat_c, dim_ff=uv_feat_c*2, num_levels=1, 
                                              num_points=self.npoints, num_heads=self.nhead)
                self.el.append(encoder_layers)
    
    def forward(self, input, frontview_features, _M_inv = None):
        projs = []
        for i in range(self.num_proj): #frontview_features的维度是torch.Size([8, 64, 90, 120])
            if i == 0:
                bev_h = self.bev_h  #bev的大小的都不一样,包含了transformer的所有方法
                bev_w = self.bev_w
            else:
                bev_h = bev_h // 2
                bev_w = bev_w // 2
            bs, c, h, w = frontview_features[i].shape
            query_embed = self.query_embeds[i].weight.unsqueeze(0).repeat(bs, 1, 1) #3d的query
            src = frontview_features[i].flatten(2).permute(0, 2, 1)
            bev_mask = torch.zeros((bs, bev_h, bev_w), device=query_embed.device).to(query_embed.dtype)
            bev_pos = self.pe[i](bev_mask).to(query_embed.dtype)
            bev_pos = bev_pos.flatten(2).permute(0, 2, 1)
            ref_2d = self.ref_2d[i].repeat(bs, 1, 1, 1).to(input.device) #input没有用上，只用了device
            ref_pnts = self.project_layers[i](_M_inv).unsqueeze(-2) #这里用到了_M_inv
            input_spatial_shapes = self.input_spatial_shapes[i].to(input.device)
            input_level_start_index = self.input_level_start_index[i].to(input.device)
            for j in range(self.num_att): #这里会经过好几次的encoder
                query_embed = self.el[i*self.num_att+j](query=query_embed, value=src, bev_pos=bev_pos, 
                                                        ref_2d = ref_2d, ref_3d=ref_pnts,#为啥这两个维度一样啊（ref_3d可能显示设置为一个维度为0）
                                                        bev_h=bev_h, bev_w=bev_w, 
                                                        spatial_shapes=input_spatial_shapes,
                                                        level_start_index=input_level_start_index) #这里就是Deformabel Attention
            query_embed = query_embed.permute(0, 2, 1).view(bs, c, bev_h, bev_w).contiguous()#强行整理为bev的feature
            projs.append(query_embed)
        return projs

    @staticmethod
    def get_reference_points(H, W, Z=8, D=4, dim='3d', bs=1, device='cuda', dtype=torch.long):
        """Get the reference points used in decoder.
        Args:
            H, W spatial shape of bev
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        # 2d to 3d reference points, need grid from M_inv
        if dim == '3d':
            raise Exception("get reference poitns 3d not supported")
            zs = torch.linspace(0.5, Z - 0.5, D, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(-1, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(D, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(D, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)

            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H  # ?
            ref_x = ref_x.reshape(-1)[None] / W  # ?
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d   