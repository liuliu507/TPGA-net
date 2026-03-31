import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmcv import cnn
from mmengine.model import BaseModule


class CPGA(nn.Module):
    """Category Prototype Guide Attention """

    def __init__(self, embed_dims, num_heads, num_classes, attn_drop_rate=.0, drop_rate=.0, qkv_bias=True,
                 mlp_ratio=4, use_memory=False, init_memory=None, norm_cfg=None, init_cfg=None):
        super(CPGA, self).__init__()

        _,self.norm_low = cnn.build_norm_layer(norm_cfg, num_features=embed_dims)
        _,self.norm_high = cnn.build_norm_layer(norm_cfg, num_features=embed_dims)
        self.cross_attn = CFTransform(embed_dims, num_heads, num_classes, attn_drop_rate,
                                      drop_rate, qkv_bias, use_memory=use_memory, init_memory=init_memory)

        _,self.norm_mlp = cnn.build_norm_layer(norm_cfg, num_features=embed_dims)
        ffn_channels = embed_dims * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dims, ffn_channels, 1, bias=True),
            nn.Conv2d(ffn_channels, ffn_channels, 3, 1, 1, groups=ffn_channels, bias=True),
            cnn.build_activation_layer(dict(type='GELU')),
            nn.Dropout(drop_rate),
            nn.Conv2d(ffn_channels, embed_dims, 1, bias=True),
            nn.Dropout(drop_rate))

    def forward(self, low, high, momentum=0.1):
        # LayerNorm 需要输入形状为 (B, H, W, C)，所以先 permute 再 permute 回来
        query = self.norm_low(low.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        key_value = self.norm_high(high.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        outs = self.cross_attn(query, key_value, momentum)

        out = outs.pop('out') + low
        out = self.mlp(self.norm_mlp(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)) + out
        outs.update({'out': out})
        return outs


class CFTransform(BaseModule):
    def __init__(self, embed_dims, num_heads, num_classes, attn_drop_rate=.0, drop_rate=.0, qkv_bias=True,
                 qk_scale=None, proj_bias=True, use_memory=False, init_memory=None, init_cfg=None):
        super(CFTransform, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims ** -0.5
        self.q = cnn.DepthwiseSeparableConvModule(embed_dims, embed_dims, 3, 1, 1,
                                                  act_cfg=None, bias=qkv_bias)
        self.kv = CFEmbedding(embed_dims, num_classes, use_memory, init_memory, qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Conv2d(embed_dims, embed_dims, 1, bias=proj_bias)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, query, key_value, momentum=0.1):
        B, C, H, W = query.shape

        # 修改：HSI和LiDAR先融合，然后生成类别原型作为key_value
        # 1. 融合HSI和LiDAR特征
        fused = (query + key_value) / 2  # 简单的平均融合

        # 2. 使用融合后的特征生成类别原型作为key_value
        fused_outs = self.kv(fused, momentum)
        fused_out = fused_outs.pop('out')  # 形状: [B, 2*C, L]

        # 3. 使用生成的原型作为key和value
        k, v = torch.chunk(fused_out, chunks=2, dim=1)  # 融合原型作为key/value: [B, C, L]

        # 4. HSI特征单独生成query（使用原始的深度可分离卷积处理）
        q = self.q(query)  # [B, C, H, W]
        q = q.reshape(B, C, -1)  # [B, C, N]

        # 重新计算head维度
        head_dims = C // self.num_heads

        # 重塑为注意力计算所需的形状
        q = q.reshape(B, self.num_heads, head_dims, -1).permute(0, 1, 3, 2)  # [B, num_heads, N, head_dims]
        k = k.reshape(B, self.num_heads, head_dims, -1).permute(0, 1, 3, 2)  # [B, num_heads, L, head_dims]
        v = v.reshape(B, self.num_heads, head_dims, -1).permute(0, 1, 3, 2)  # [B, num_heads, L, head_dims]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, L]
        attn = torch.max(attn, -1, keepdim=True)[0].expand_as(attn) - attn  # stable training
        attn = F.softmax(attn, dim=-1)
        outs = {}
        outs.update({'attn': torch.mean(attn, dim=1, keepdim=False)})
        attn = self.attn_drop(attn)

        # 计算输出
        out = (attn @ v).transpose(-2, -1)  # [B, num_heads, head_dims, N]
        out = out.reshape(B, C, H, W)  # [B, C, H, W]

        out = self.proj_drop(self.proj(out))
        outs.update({'out': out})
        return outs


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class CFEmbedding(BaseModule):
    def __init__(self, embed_dims, num_classes, use_memory, init_memory=None, kv_bias=True,
                 num_groups=4, init_cfg=None):
        super(CFEmbedding, self).__init__(init_cfg)
        if use_memory:
            if init_memory is None:  # random init
                std = 1. / ((num_classes * embed_dims) ** 0.5)
                memory = torch.empty(1, num_classes, embed_dims).normal_(0, std)
            else:  # pretrained init
                memory = torch.tensor(np.load(init_memory), dtype=torch.float)[:, :embed_dims].unsqueeze(0)
            memory = F.normalize(memory, dim=2, p=2)
            self.register_buffer('memory', memory)

        self.mask_learner = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 1, groups=num_groups, bias=False),
            nn.Conv2d(embed_dims, num_classes, 1, bias=False))
        self.align_conv = nn.Conv2d(embed_dims, embed_dims, 1, groups=num_groups, bias=False)
        self.cf_embed = nn.Linear(embed_dims, embed_dims * 2, bias=kv_bias)

    @torch.no_grad()
    def _update_memory(self, cf_feat, momentum=0.1):
        cf_feat = cf_feat.mean(dim=0, keepdim=True)
        cf_feat = reduce_mean(cf_feat)  # sync across GPUs
        cf_feat = F.normalize(cf_feat, dim=2, p=2)
        self.memory = (1.0 - momentum) * self.memory + momentum * cf_feat

    def forward(self, x, momentum=0.1):
        mask = self.mask_learner(x)
        outs = {'mask': mask}
        mask = mask.reshape(mask.size(0), mask.size(1), -1)  # B x L x N
        mask = F.softmax(mask, dim=-1)

        x = self.align_conv(x)
        x = x.reshape(x.size(0), x.size(1), -1)  # B x C x N
        cf_feat = mask @ x.transpose(-2, -1)  # category feature: B x L x C

        if hasattr(self, 'memory'):
            memory = self.memory.expand(cf_feat.size(0), -1, -1)
            if self.training:
                self._update_memory(cf_feat, momentum)
            cf_feat = (1.0 - momentum) * cf_feat + momentum * memory

        out = self.cf_embed(cf_feat)
        outs.update({'out': out.transpose(-2, -1)})  # B x C x L, used as key/value
        return outs
