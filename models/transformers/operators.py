from collections import OrderedDict
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange as o_rearrange
from einops.layers.torch import Rearrange

from utils.utils import to_2tuple
from timm.models.layers import DropPath, trunc_normal_

BATCHNORM = nn.SyncBatchNorm


def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()


class UpEmbed(nn.Module):
    def __init__(self, patch_size=7, in_chans=3, embed_dim=64, stride=4, padding=2):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                      padding=padding, stride=stride, bias=False, dilation=padding),
            BATCHNORM(embed_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size,
                      padding=padding, stride=stride, bias=False, dilation=padding),
            BATCHNORM(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self,
                 fea_no,
                 dim_in,
                 query_dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 q_method='dw_bn',
                 kv_method='dw_bn',
                 kernel_size_q=3,
                 kernel_size_kv=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 **kwargs):
        super().__init__()
        self.stride_kv = int(stride_kv)
        self.stride_q = int(stride_q)
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.fea_no = int(fea_no)

        self.conv_proj_q = self._build_single_projection(
            query_dim_in, kernel_size_q, padding_q, stride_q, q_method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, kernel_size_kv, padding_kv, stride_kv, kv_method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, kernel_size_kv, padding_kv, stride_kv, kv_method
        )
        self.restore_proj_q = self._build_restore_projection(dim_out)

        self.proj_q = nn.Linear(query_dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_single_projection(self, dim_in, kernel_size, padding, stride, method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size,
                                   padding=padding, stride=stride,
                                   bias=False, groups=dim_in)),
                ('bn', BATCHNORM(dim_in)),
                ('rearrange', Rearrange('b c h w -> b (h w) c'))
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(kernel_size=kernel_size, padding=padding,
                                     stride=stride, ceil_mode=True)),
                ('rearrange', Rearrange('b c h w -> b (h w) c'))
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError(f'Unknown method ({method})')
        return proj

    def _build_restore_projection(self, dim_in):
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, stride=1, groups=dim_in))
        ]))

    def _build_projection(self, dim_in, kernel_size, padding, stride, method):
        if method == 'dw_bn':
            proj = [nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size,
                                   padding=padding, stride=stride,
                                   bias=False, groups=dim_in)),
                ('bn', BATCHNORM(dim_in)),
                ('rearrange', Rearrange('b c h w -> b (h w) c')),
            ])) for _ in range(self.fea_no)]
            return nn.ModuleList(proj)
        if method == 'avg':
            proj = [nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(kernel_size=kernel_size, padding=padding,
                                     stride=stride, ceil_mode=True)),
                ('rearrange', Rearrange('b c h w -> b (h w) c')),
            ])) for _ in range(self.fea_no)]
            return nn.ModuleList(proj)
        if method == 'linear':
            return None
        raise ValueError(f'Unknown method ({method})')

    def split_x(self, x, h, w):
        h = int(h); w = int(w)
        res = int(h * w)
        x_list = []
        for i in range(self.fea_no):
            _x = rearrange(x[:, res * i:res * (i + 1), :], 'b (h w) c -> b c h w', h=h, w=w)
            x_list.append(_x)
        return x_list

    def forward_conv(self, x, query, h, w):
        h = int(h); w = int(w)
        x_list = self.split_x(x, h, w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(query)
        else:
            q = rearrange(query, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k_list = [self.conv_proj_k[i](x_list[i]) for i in range(self.fea_no)]
            k = torch.cat(k_list, dim=1)
        else:
            k_list = [rearrange(xx, 'b c h w -> b (h w) c') for xx in x_list]
            k = torch.cat(k_list, dim=1)

        if self.conv_proj_v is not None:
            v_list = [self.conv_proj_v[i](x_list[i]) for i in range(self.fea_no)]
            v = torch.cat(v_list, dim=1)
        else:
            v_list = [rearrange(xx, 'b c h w -> b (h w) c') for xx in x_list]
            v = torch.cat(v_list, dim=1)

        return q, k, v

    def forward(self, x, query, h, w):
        h = int(h); w = int(w)
        _, _, H0, W0 = query.shape
        H0 = int(H0); W0 = int(W0)

        q, k, v = self.forward_conv(x, query, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        out = rearrange(out, 'b h t d -> b t (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)

        Hs = int(H0 // self.stride_q)
        Ws = int(W0 // self.stride_q)

        out = rearrange(out, 'b (h w) c -> b c h w', h=Hs, w=Ws)
        out = self.restore_proj_q(F.interpolate(out, size=(H0, W0), mode='bilinear', align_corners=False))
        out = rearrange(out, 'b c h w -> b (h w) c', h=H0, w=W0)

        return out


class PrecisionFieldEstimator(nn.Module):
    def __init__(self, num_rules: int = 8):
        super().__init__()
        self.num_rules = int(num_rules)

        self.centers = nn.Parameter(torch.randn(self.num_rules, 2) * 0.2)
        self.log_sigma = nn.Parameter(torch.zeros(self.num_rules, 2))

        self.conseq = nn.Parameter(torch.zeros(self.num_rules, 3))
        nn.init.constant_(self.conseq[:, 0],  3.0)
        nn.init.constant_(self.conseq[:, 1], -2.0)
        nn.init.constant_(self.conseq[:, 2],  0.0)

    @staticmethod
    def _cos_sim(a, b, eps=1e-6):
        a = F.normalize(a, dim=1, eps=eps)
        b = F.normalize(b, dim=1, eps=eps)
        return (a * b).sum(dim=1, keepdim=True)

    @staticmethod
    def _tv(x):
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return dx.mean(dim=1, keepdim=True) + dy.mean(dim=1, keepdim=True)

    def forward(self, E, G):
        sim = self._cos_sim(E, G)
        tv = self._tv(E)
        z = torch.cat([sim, tv], dim=1)

        c = self.centers.view(1, self.num_rules, 2, 1, 1)
        s = F.softplus(self.log_sigma).view(1, self.num_rules, 2, 1, 1) + 1e-6
        zz = z.unsqueeze(1)

        dist2 = ((zz - c) / s).pow(2).sum(dim=2)
        mu = torch.exp(-0.5 * dist2)
        mu = mu / (mu.sum(dim=1, keepdim=True) + 1e-6)

        a_sim = self.conseq[:, 0].view(1, self.num_rules, 1, 1)
        a_tv = self.conseq[:, 1].view(1, self.num_rules, 1, 1)
        b = self.conseq[:, 2].view(1, self.num_rules, 1, 1)

        log_alpha_rules = a_sim * sim + a_tv * tv + b
        log_alpha = (mu * log_alpha_rules).sum(dim=1, keepdim=True)

        alpha = F.softplus(log_alpha) + 1e-6
        return alpha, log_alpha, mu


class PosteriorBridgeOperator(nn.Module):
    def forward(self, G, E_list, w0, w_list, eps=1e-6):
        num = w0 * G
        den = w0
        for Et, wt in zip(E_list, w_list):
            num = num + wt * Et
            den = den + wt
        return num / (den + eps)


class ContractiveDispatchOperator(nn.Module):
    def __init__(self, channels: int, use_w: bool = True):
        super().__init__()
        self.use_w = bool(use_w)
        in_ch = int(channels) + (1 if self.use_w else 0)
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, channels, 1, bias=True),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[0].bias, -2.0)
        self.eta = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, ref, w=None):
        if self.use_w:
            if w is None:
                raise ValueError("CDO requires w when use_w=True")
            g = self.gate(torch.cat([ref, w], dim=1))
        else:
            g = self.gate(ref)
        eta = torch.sigmoid(self.eta)
        return x + (eta * g) * (ref - x)


class PosteriorBridgeBlock(nn.Module):
    def __init__(self,
                 task_no,
                 dim_in,
                 query_dim_in,
                 dim_out,
                 query_dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 channel_restore_opt=False,
                 **kwargs):
        super().__init__()

        self.task_no = int(task_no)
        self.embed_dim = int(dim_in // self.task_no)
        self.query_dim_in = int(query_dim_in)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_q = BATCHNORM(self.query_dim_in)

        attn_kwargs = dict(
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            q_method=kwargs['q_method'],
            kv_method=kwargs['kv_method'],
            kernel_size_q=kwargs['kernel_size_q'],
            kernel_size_kv=kwargs['kernel_size_kv'],
            stride_kv=kwargs['stride_kv'],
            stride_q=kwargs['stride_q'],
            padding_kv=kwargs['padding_kv'],
            padding_q=kwargs['padding_q'],
        )

        self.task_attn = nn.ModuleList([
            CrossAttention(
                fea_no=1,
                dim_in=self.embed_dim,
                query_dim_in=self.query_dim_in,
                dim_out=self.query_dim_in,
                num_heads=num_heads,
                **attn_kwargs
            ) for _ in range(self.task_no)
        ])

        self.pfe = PrecisionFieldEstimator(num_rules=8)
        self.pbo = PosteriorBridgeOperator()

        self.prior_logit = nn.Conv2d(self.query_dim_in, 1, 1, bias=True)
        nn.init.zeros_(self.prior_logit.weight)
        nn.init.constant_(self.prior_logit.bias, 1.0)

        self.eta_refine = nn.Parameter(torch.tensor(-1.0))

        self.norm_tok = norm_layer(self.query_dim_in)
        dim_mlp_hidden = int(self.query_dim_in * mlp_ratio)
        self.mlp = Mlp(self.query_dim_in, dim_mlp_hidden, self.query_dim_in, act_layer=act_layer, drop=drop)

        self.last_w_list: Optional[List[torch.Tensor]] = None

    def forward(self, x_list: List[torch.Tensor], backbone_feat: torch.Tensor) -> torch.Tensor:
        h, w = x_list[0].shape[2:]
        h = int(h); w = int(w)

        B, Cq, H0, W0 = backbone_feat.shape
        H0 = int(H0); W0 = int(W0)

        G = self.norm_q(backbone_feat)
        w0 = F.softplus(self.prior_logit(G)) + 1e-6

        E_list: List[torch.Tensor] = []
        w_list: List[torch.Tensor] = []

        for t in range(self.task_no):
            xt = x_list[t]
            xt_tok = rearrange(xt, 'b c hh ww -> b (hh ww) c')                

            Etok = self.task_attn[t](xt_tok, G, h, w)

            Emap = rearrange(Etok, 'b (hh ww) c -> b c hh ww', hh=H0, ww=W0).contiguous()

            wt, _, _ = self.pfe(Emap, G)
            E_list.append(Emap)
            w_list.append(wt)

        S = self.pbo(G, E_list, w0=w0, w_list=w_list)

        eta = torch.sigmoid(self.eta_refine)
        refined = G + eta * (S - G)

        tok = rearrange(refined, 'b c hh ww -> b (hh ww) c')
        tok = tok + self.drop_path(self.mlp(self.norm_tok(tok)))
        refined = rearrange(tok, 'b (hh ww) c -> b c hh ww', hh=H0, ww=W0).contiguous()

        self.last_w_list = w_list
        return refined


class TaskRefinementBlock(nn.Module):
    def __init__(self,
                 dim_in,
                 ref_dim,
                 h_dim,
                 dim_out,
                 dilated_ratio,
                 drop_path=0.):
        super().__init__()
        dim_in = int(dim_in)
        ref_dim = int(ref_dim)

        self.conv_layer = nn.Sequential(
            nn.Conv2d(dim_in + ref_dim, h_dim, 1, 1, 0),
            BATCHNORM(h_dim),
            nn.GELU(),

            nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=dilated_ratio,
                      stride=1, bias=False, dilation=dilated_ratio, groups=h_dim),
            BATCHNORM(h_dim),
            nn.GELU(),

            nn.Conv2d(h_dim, dim_out, 1, 1, 0)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, res):
        return res + self.drop_path(self.conv_layer(x))


class TaskRefinementStage(nn.Module):
    def __init__(self,
                 task_no,
                 dim_in,
                 dim_out,
                 depth,
                 dilated_ratio,
                 squeeze_ratio,
                 drop_path_rate=0.,
                 ref_dim: Optional[int] = None):
        super().__init__()

        self.task_no = int(task_no)
        self.dim_in = int(dim_in)
        self.ref_dim = int(ref_dim) if ref_dim is not None else int(dim_in)

        re_blocks = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for j in range(depth):
            h_dim = self.dim_in // squeeze_ratio[j]
            re_blocks.append(
                nn.ModuleList([
                    TaskRefinementBlock(dim_in=self.dim_in, ref_dim=self.ref_dim, h_dim=h_dim,
                                    dim_out=dim_out, dilated_ratio=dilated_ratio[j], drop_path=dpr[j])
                    for _ in range(self.task_no)
                ])
            )
        self.re_blocks = nn.ModuleList(re_blocks)
        assert depth % 3 == 0

        self.dispatch = nn.ModuleList([
            ContractiveDispatchOperator(channels=self.dim_in, use_w=True) for _ in range(self.task_no)
        ])

        if self.ref_dim != self.dim_in:
            self.ref_to_task = nn.Conv2d(self.ref_dim, self.dim_in, 1, 1, 0)
        else:
            self.ref_to_task = None

    def _split_x_concatenation(self, x_list, ref_feat, w_list=None):
        _, _, h, w = x_list[0].shape
        h = int(h); w = int(w)
        ref_up = F.interpolate(ref_feat, size=(h, w), mode='bilinear', align_corners=False)

        ref_for_dispatch = self.ref_to_task(ref_up) if self.ref_to_task is not None else ref_up

        added_x_list = []
        for i, x in enumerate(x_list):
            if w_list is not None:
                wi = F.interpolate(w_list[i], size=(h, w), mode='bilinear', align_corners=False)
                x_inj = self.dispatch[i](x, ref_for_dispatch, wi)
            else:
                x_inj = x

            ref_x = torch.cat([x_inj, ref_up], dim=1)
            added_x_list.append(ref_x)
        return added_x_list

    def forward(self, x_list, refined_feature, w_list=None):
        seq_out_list = []
        for j, blk in enumerate(self.re_blocks):
            key_x_list = self._split_x_concatenation(x_list, refined_feature, w_list=w_list)
            out_list = []
            for i in range(len(x_list)):
                out_list.append(blk[i](key_x_list[i], x_list[i]))
            if j % 3 == 2:
                seq_out_list.append(out_list)
            x_list = out_list
        return x_list, seq_out_list


class B3Stage(nn.Module):
    def __init__(self,
                 p,
                 stage_idx,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=[],
                 back_out_chans=[],
                 depth=1,
                 refine_depth=3,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 squeeze_ratio=[2, 2, 2],
                 dilated_ratio=[1, 2, 5],
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 channel_restore_opt=False,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        assert depth == 1
        self.stage_idx = int(stage_idx)
        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim[stage_idx])
        self.task_no = len(p.TASKS.NAMES)

        self.back_feature_embed = nn.Sequential(
            nn.Conv2d(in_channels=back_out_chans[stage_idx],
                      out_channels=embed_dim[stage_idx],
                      kernel_size=1,
                      stride=1),
            BATCHNORM(embed_dim[stage_idx]),
            nn.ReLU(inplace=True),
        )

        if patch_size == 0:
            self.patch_embed = None
        else:
            self.patch_embed = nn.ModuleList([
                UpEmbed(patch_size=patch_size,
                        in_chans=in_chans,
                        stride=patch_stride,
                        padding=patch_padding,
                        embed_dim=embed_dim[stage_idx])
                for _ in range(self.task_no)
            ])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            PosteriorBridgeBlock(
                dim_in=embed_dim[stage_idx] * self.task_no,
                query_dim_in=embed_dim[stage_idx],
                dim_out=embed_dim[stage_idx] * self.task_no,
                query_dim_out=back_out_chans[stage_idx],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[0],
                act_layer=act_layer,
                norm_layer=norm_layer,
                channel_restore_opt=channel_restore_opt,
                **kwargs
            )
        ])

        self.refine_depth = int(refine_depth)
        if self.refine_depth > 0:
            self.reconv = TaskRefinementStage(
                task_no=self.task_no,
                dim_in=embed_dim[stage_idx],
                dim_out=embed_dim[stage_idx],
                depth=refine_depth,
                dilated_ratio=dilated_ratio,
                squeeze_ratio=squeeze_ratio,
                drop_path_rate=drop_path_rate,
                ref_dim=embed_dim[stage_idx],
            )

        if stage_idx > 0:
            self.reduce_channel = nn.Conv2d(embed_dim[stage_idx - 1], embed_dim[stage_idx], 1, 1, 0)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, BATCHNORM)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, BATCHNORM)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _split_x_addition(self, x_list, ref_feat):
        _, _, h, w = x_list[0].shape
        h = int(h); w = int(w)
        out = []
        for x in x_list:
            out.append(x + F.interpolate(ref_feat, size=(h, w), mode='bilinear', align_corners=False))
        return out

    def forward(self, x_list, back_fea, last_back_feat):
        if self.patch_embed is not None:
            key_x_list = [self.patch_embed[i](x_list[i]) for i in range(self.task_no)]

            if self.stage_idx == 1:
                back_fea_at_scale = self.back_feature_embed(back_fea[2])
            elif self.stage_idx == 2:
                back_fea_at_scale = self.back_feature_embed(back_fea[1])
            elif self.stage_idx == 3:
                back_fea_at_scale = self.back_feature_embed(back_fea[0])
            else:
                back_fea_at_scale = self.back_feature_embed(back_fea[3])
        else:
            key_x_list = x_list
            back_fea_at_scale = self.back_feature_embed(back_fea[3])

        refined_feature = None
        w_list = None

        for blk in self.blocks:
            if last_back_feat is not None:
                _, _, H0, W0 = back_fea_at_scale.shape
                H0 = int(H0); W0 = int(W0)
                last_back_feat = self.reduce_channel(last_back_feat)
                refined_feature = blk(key_x_list, back_fea_at_scale) + F.interpolate(
                    last_back_feat, size=(H0, W0), mode='bilinear', align_corners=False
                )
            else:
                refined_feature = blk(key_x_list, back_fea_at_scale)

            w_list = getattr(blk, "last_w_list", None)

        if self.refine_depth > 0:
            key_x_list, _ = self.reconv(key_x_list, refined_feature, w_list=w_list)
        else:
            key_x_list = self._split_x_addition(key_x_list, refined_feature)

        return refined_feature, key_x_list


class B3FeaturePropagator(nn.Module):
    def __init__(self,
                 p,
                 in_chans=3,
                 num_classes=1000,
                 back_channels=[144, 288, 576],
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        assert spec is not None, "spec must be provided"
        self.p = p

        self.all_tasks = p.TASKS.NAMES
        self.task_no = len(self.all_tasks)

        self.num_stages = spec['NUM_STAGES']
        self.embed_dim_list = spec['DIM_EMBED']
        self.query_out_chans = [back_channels[len(back_channels) - 1 - i] for i in range(len(back_channels))]

        mt_in_chans = in_chans
        self.norm_mts = nn.ModuleList()
        self.mt_embed_dims: List[int] = []
        target_channel = in_chans

        self.redu_chan = nn.ModuleList()
        self.trfl_stages = nn.ModuleList()

        for i in range(self.num_stages):
            cur_mt_embed_dim = spec['DIM_EMBED'][i]
            kwargs = {
                'task_no': self.task_no,
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'depth': 1,
                'refine_depth': spec['REFINE_DEPTH'][i],
                'dilated_ratio': spec['DILATED_RATIO'][i],
                'squeeze_ratio': spec['SQUEEZE_RATIO'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': 0,
                'attn_drop_rate': 0,
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'q_method': spec['Q_PROJ_METHOD'][i],
                'kv_method': spec['KV_PROJ_METHOD'][i],
                'kernel_size_q': spec['KERNEL_Q'][i],
                'kernel_size_kv': spec['KERNEL_KV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }

            stage = B3Stage(
                p=p,
                stage_idx=i,
                in_chans=mt_in_chans,
                back_out_chans=self.query_out_chans,
                embed_dim=self.embed_dim_list,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            self.trfl_stages.append(stage)

            mt_in_chans = cur_mt_embed_dim
            self.norm_mts.append(norm_layer(mt_in_chans * self.task_no))
            self.mt_embed_dims.append(mt_in_chans)

            _redu_chan = nn.ModuleList([nn.Conv2d(mt_in_chans, target_channel, 1) for _ in range(self.task_no)])
            self.redu_chan.append(_redu_chan)

        self.mt_embed_dim = target_channel

        self.mt_proj = nn.ModuleDict()
        for task in self.all_tasks:
            self.mt_proj[task] = nn.Sequential(
                nn.Conv2d(self.mt_embed_dim, self.mt_embed_dim, 3, padding=1),
                BATCHNORM(self.mt_embed_dim),
                nn.ReLU(True)
            )
            trunc_normal_(self.mt_proj[task][0].weight, std=0.02)

        self.mix_proj = nn.ModuleDict()
        for t in self.all_tasks:
            _mix_channel = spec['ori_embed_dim'] + p.TASKS.NUM_OUTPUT[t]
            self.mix_proj[t] = nn.Sequential(nn.Conv2d(_mix_channel, in_chans, 1))

    def forward(self, x_dict: Dict[str, torch.Tensor], inter_pred: Dict[str, torch.Tensor], back_fea):
        x_list = []
        for t in self.all_tasks:
            _x = torch.cat([x_dict[t], inter_pred[t]], dim=1)
            _x = self.mix_proj[t](_x)
            x_list.append(_x)

        h, w = self.p.mtt_resolution
        h = int(h); w = int(w)
        th = int(h * (2 ** (self.num_stages - 1)) * 2)
        tw = int(w * (2 ** (self.num_stages - 1)) * 2)

        multi_scale_task_feature = {t: 0 for t in self.all_tasks}
        refined_feat = None

        for i in range(self.num_stages):
            refined_feat, x_list = self.trfl_stages[i](x_list, back_fea, refined_feat)

            nh = int(h * (2 ** i))
            nw = int(w * (2 ** i))

            _x_list = [
                F.interpolate(_x, size=(nh, nw), mode='bilinear', align_corners=False) if i == 0 else _x
                for _x in x_list
            ]
            _x_list = [rearrange(_x, 'b c hh ww -> b (hh ww) c') for _x in _x_list]
            x = torch.cat(_x_list, dim=2)
            x = self.norm_mts[i](x)
            x = rearrange(x, 'b (hh ww) c -> b c hh ww', hh=nh, ww=nw).contiguous()

            for ii, task in enumerate(self.all_tasks):
                mt_embed_dim = int(self.mt_embed_dims[i])
                task_x = x[:, mt_embed_dim * ii: mt_embed_dim * (ii + 1), :, :]
                if i > 0:
                    task_x = self.redu_chan[i][ii](task_x)
                task_x = F.interpolate(task_x, size=(th, tw), mode='bilinear', align_corners=False)
                multi_scale_task_feature[task] += task_x

        out_dict = {}
        for task in self.all_tasks:
            out_dict[task] = self.mt_proj[task](multi_scale_task_feature[task])

        return out_dict

