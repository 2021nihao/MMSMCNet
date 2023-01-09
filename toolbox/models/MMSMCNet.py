import math

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_
from torch.nn.parameter import Parameter
from torch.nn import init

from backbone.SegFormer_master.mmseg.models.backbones.mix_transformer import mit_b3


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1, bn=True, relu=True):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        # padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                              bias=False if bn else True)
        self.bn = bn
        if bn:
            self.bnop = nn.BatchNorm2d(out_planes)
        self.relu = relu
        if relu:
            # self.reluop = nn.ReLU6(inplace=True)
            self.reluop = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bnop(x)
        if self.relu:
            x = self.reluop(x)
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=104, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super(OverlapPatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class ExternalAttention(nn.Module):
  
    def __init__(self, d_model,S=32):
        super().__init__()
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model

        return out

class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class PE_MHA(nn.Module):
    def __init__(self, img_size=104, patch_size=3, stride=1, in_chans=64, d_model=None,
                 embed_dim=64, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0, sr_ratio=1, drop_rate=0):
        super(PE_MHA, self).__init__()
        self.laynorm1 = nn.LayerNorm(embed_dim)
        self.laynorm2 = nn.LayerNorm(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patchembed = OverlapPatchEmbed(img_size=img_size, patch_size=patch_size, stride=stride, in_chans=in_chans,
                                              embed_dim=embed_dim)
        self.attention = Attention(embed_dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop_rate, sr_ratio=sr_ratio)
        self.external_att = ExternalAttention(d_model=d_model)

    def forward(self, x):
        B = x.shape[0]
        x,  H, W = self.patchembed(x)
        x = self.laynorm1(x)
        # x = self.attention(x, H, W )
        x = self.external_att(x)
        x = self.pos_drop(x)
        x = self.laynorm2(x)
        out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return out

class Fusion(nn.Module):
    def __init__(self, img_size, patch_size, stride, in_chans, h, w, d_model):
        super(Fusion, self).__init__()
        self.pe_mha_rgb = PE_MHA(img_size, patch_size, stride, in_chans, d_model=d_model,
                 embed_dim=64, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0, sr_ratio=1, drop_rate=0)
        self.pe_mha_t = PE_MHA(img_size, patch_size, stride, in_chans, d_model=d_model,
                                 embed_dim=64, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0, sr_ratio=1,
                                 drop_rate=0)
        self.GloablAvgPool_rgb = nn.AdaptiveAvgPool2d(output_size=(h//2, w//2))
        self.GloablAvgPool_t = nn.AdaptiveAvgPool2d(output_size=(h//2, w//2))

        self.GloablAvgPool_rgb_4 = nn.AdaptiveAvgPool2d(output_size=(h , w ))
        self.GloablAvgPool_t_4 = nn.AdaptiveAvgPool2d(output_size=(h , w))

        self.conv1_rgb1 = nn.Conv2d(in_channels=64, out_channels=64//4, kernel_size=1, stride=1, padding=0)
        self.conv1_t1 = nn.Conv2d(in_channels=64, out_channels=64//4, kernel_size=1, stride=1, padding=0)
        self.conv1_rgb2 = nn.Conv2d(in_channels=64//4, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1_t2 = nn.Conv2d(in_channels=64//4, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.cbl_1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU()
                                      )
        self.cbl_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU()
                                   )
        self.cbl_3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU()
                                   )
        self.cbl_3_4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU()
                                   )
        self.cbl_4 = nn.Sequential(nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU()
                                   )
        self.cbl_4_1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU()
                                   )
        self.conv3_add = nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3_mul = nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.conv1_add = nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_mul = nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, t, f, NO):
        rgbt_add = rgb + t  #c*h*w
        rgbt_mul = torch.mul(rgb, t)  #c*h*w
        # print(rgbt_add.shape)

        rgbt_add_former = self.pe_mha_rgb(rgbt_add)
        rgbt_mul_former = self.pe_mha_t(rgbt_mul)
        # print(rgbt_mul_former.shape)

        if NO == 4:
            rgbt_add_2 = self.conv1_add(rgbt_add)
            rgbt_mul_2 = self.conv1_mul(rgbt_mul)

        else:
            rgbt_add_2 = self.conv3_add(rgbt_add)
            rgbt_mul_2 = self.conv3_mul(rgbt_mul)
        # print(rgbt_add_2.shape , rgbt_add_former.shape)
        rgbt_add_former = rgbt_add_2 + rgbt_add_former
        rgbt_mul_former = rgbt_mul_2 + rgbt_mul_former

        if NO == 4:
            rgbt_add_avgpool = self.GloablAvgPool_rgb_4(rgbt_add_2)
            rgbt_add_conv = self.conv1_rgb1(rgbt_add_avgpool)
            rgbt_add_conv = self.conv1_rgb2(rgbt_add_conv)
            # print(rgbt_add_former.shape, rgbt_add_conv.shape)
            rgbt_add_conv = torch.mul(rgbt_add_former, rgbt_add_conv)

            rgbt_mul_avgpool = self.GloablAvgPool_t_4(rgbt_mul_2)
            rgbt_mul_conv = self.conv1_t1(rgbt_mul_avgpool)
            rgbt_mul_conv = self.conv1_t2(rgbt_mul_conv)
            rgbt_mul_conv = torch.mul(rgbt_mul_former, rgbt_mul_conv)

        else:
            rgbt_add_avgpool = self.GloablAvgPool_rgb(rgbt_add_2)
            rgbt_add_conv = self.conv1_rgb1(rgbt_add_avgpool)
            rgbt_add_conv = self.conv1_rgb2(rgbt_add_conv)
            # print(rgbt_add_former.shape, rgbt_add_conv.shape)
            rgbt_add_conv = torch.mul(rgbt_add_former, rgbt_add_conv)

            rgbt_mul_avgpool = self.GloablAvgPool_t(rgbt_mul_2)
            rgbt_mul_conv = self.conv1_t1(rgbt_mul_avgpool)
            rgbt_mul_conv = self.conv1_t2(rgbt_mul_conv)
            rgbt_mul_conv = torch.mul(rgbt_mul_former, rgbt_mul_conv)

        rgbt_add_conv = self.cbl_1(rgbt_add_conv)
        rgbt_mul_conv = self.cbl_2(rgbt_mul_conv)
        if NO==1:
            rgbt = torch.cat((rgbt_add_conv, rgbt_mul_conv), dim=1)
            rgbt = self.cbl_4_1(rgbt)
        elif NO==4:
            f = self.cbl_3_4(f)
            rgbt = torch.cat((rgbt_add_conv, rgbt_mul_conv, f), dim=1)
            rgbt = self.cbl_4(rgbt)

        else:
            f = self.cbl_3(f)
            rgbt = torch.cat((rgbt_add_conv, rgbt_mul_conv, f), dim=1)
            rgbt = self.cbl_4(rgbt)

        return rgbt

class ChannelAttention(nn.Module):
    # def __init__(self, in_planes, ratio=16):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # self.fc1 = nn.Conv2d(in_planes, in_planes / 16, 1, bias=False)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.LeakyReLU()
        # self.fc2 = nn.Conv2d(in_planes / 16, in_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)   #输出格式为 1*1*c

class Channel_Max_Pooling(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Channel_Max_Pooling, self).__init__()
        self.max_pooling = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride
        )
    def forward(self, x):
        # print('Input_Shape:', x.shape)  # (batch_size, chs, h, w)
        x = x.transpose(1, 3)  # (batch_size, w, h, chs)
        # print('Transpose_Shape:', x.shape)
        x = self.max_pooling(x)
        # print('Transpose_MaxPooling_Shape:', x.shape)
        out = x.transpose(1, 3)
        # print('Output_Shape:', out.shape)
        return out

class BoundaryPart(nn.Module):
    def __init__(self):
        super(BoundaryPart, self).__init__()
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8x = nn.UpsamplingBilinear2d(scale_factor=8)
        self.cbl1 = ConvBNReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1)
        self.cbl2 = ConvBNReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1)
        self.cbl3 = ConvBNReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1)
        self.cbl4 = ConvBNReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1)
        self.cbl5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1),
                                  ConvBNReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1))
        self.cbl6 = ConvBNReLU(in_planes=256, out_planes=64, kernel_size=3, stride=1)
        self.cbl_d1 = ConvBNReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1,dilation=1)
        self.cbl_d3 = ConvBNReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1, dilation=3)
        self.cbl_d6 = ConvBNReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1, dilation=6)
        self.cbl_d12 = ConvBNReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1, dilation=12)

    def forward(self, NO, e1, e2, di):
        if NO==1:
            e2 = self.up2x(e2)
            e2 = self.cbl1(e2)
            e1 = self.cbl2(e1)
            di = self.up4x(di)
            di = self.cbl3(di)
        elif NO==2:
            e2 = self.up4x(e2)
            e2 = self.cbl1(e2)
            e1 = self.up2x(e1)
            e1 = self.cbl2(e1)
            di = self.up2x(di)
            di = self.cbl3(di)
        elif NO==3:
            e2 = self.up8x(e2)
            e2 = self.cbl1(e2)
            e1 = self.up4x(e1)
            e1 = self.cbl2(e1)
            di = self.up2x(di)
            di = self.cbl3(di)

        e12 = torch.mul(e1, e2)
        e12 = self.cbl4(e12)
        edi = torch.cat((e1, di), dim=1)
        edi = self.cbl5(edi)
        f = e12 + edi
        f1 = self.cbl_d1(f)
        f3 = self.cbl_d3(f)
        f6 = self.cbl_d6(f)
        f12 = self.cbl_d12(f)
        f_out = torch.cat((f1, f3, f6, f12), dim=1)
        out = self.cbl6(f_out)

        return out

class SemanticPart(nn.Module):
    def __init__(self):
        super(SemanticPart, self).__init__()
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8x = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up16x = nn.UpsamplingBilinear2d(scale_factor=16)
        self.cbl1 = ConvBNReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1)
        self.cbl2 = ConvBNReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1)
        self.cbl3 = ConvBNReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1)
        self.cbl4 = ConvBNReLU(in_planes=128, out_planes=64, kernel_size=3, stride=1)
        self.external_attention = External_attention(c=128)
        self.CMP = Channel_Max_Pooling((1, 2), (1, 2))

    def forward(self, NO, e3, e4, di):
        if NO==1:
            di = self.up4x(di)
            di = self.cbl1(di)
            e3 = self.up4x(e3)
            e3 = self.cbl2(e3)
            e4 = self.up4x(e4)

        elif NO==2:
            di = self.up2x(di)
            di = self.cbl1(di)
            e3 = self.up8x(e3)
            e3 = self.cbl2(e3)
            e4 = self.up8x(e4)
        elif NO==3:
            di = self.up2x(di)
            di = self.cbl1(di)
            e3 = self.up16x(e3)
            e3 = self.cbl2(e3)
            e4 = self.up16x(e4)

        e34 = self.cbl3(e4 + e3)
        e34_cmp = self.CMP(torch.cat((e3, e4), dim=1))
        e34 = torch.mul(e34, e34_cmp)
        f_out = torch.cat((di, e34), dim=1)
        out = self.external_attention(f_out)
        out = self.cbl4(out)

        return out

class EnhancePart(nn.Module):
    def __init__(self):
        super(EnhancePart, self).__init__()
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.cbl1 = ConvBNReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1)
        self.cbl2 = ConvBNReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1)
        self.cbl3 = ConvBNReLU(in_planes=192, out_planes=64, kernel_size=3, stride=1)

    def forward(self, NO, bi, si, dj):
        # bi_s = torch.sigmoid(bi)
        # si = self.cbl1(si)
        # sbi = torch.mul((1-bi_s), si) + bi
        if NO==1:
            dj = self.up4x(dj)
            dj = self.cbl2(dj)
        else:
            dj = self.up2x(dj)
            dj = self.cbl2(dj)
        bi_s = torch.sigmoid(bi)
        si = self.cbl1(si)
        sbi = torch.mul((1 - bi_s), si) + bi
        f_out = torch.cat((si, dj, sbi), dim=1)
        out = self.cbl3(f_out)

        return out

class nation(nn.Module):
    def __init__(self):
        super(nation, self).__init__()

        self.rgb = mit_b3()  # .load_state_dict(torch.load("/home/noone/桌面/sp_vgg_new/backbone/SegFormer_master/weight/mit_b1.pth"))
        self.t = mit_b3()  # .load_state_dict(torch.load("/home/noone/桌面/sp_vgg_new/backbone/SegFormer_master/weight/mit_b1.pth"))

        if self.training:
            self.rgb.load_state_dict(torch.load("/home/noone/桌面/sp_vgg_new/backbone/SegFormer_master/weight/mit_b3.pth"), strict=False)
            self.t.load_state_dict(torch.load("/home/noone/桌面/sp_vgg_new/backbone/SegFormer_master/weight/mit_b3.pth"), strict=False)
          
        # # 384*384_convnext
        # self.fusion1 = Fusion(img_size=96, patch_size=3, stride=2, in_chans=96, h=96, w=96)
        # self.fusion2 = Fusion(img_size=48, patch_size=3, stride=2, in_chans=192, h=48, w=48)
        # self.fusion3 = Fusion(img_size=24, patch_size=3, stride=2, in_chans=384, h=24, w=24)
        # self.fusion4 = Fusion(img_size=12, patch_size=3, stride=1, in_chans=768, h=12, w=12)

        # ##416*416_segformer
        self.fusion1 = Fusion(img_size=104, patch_size=3, stride=2, in_chans=64, h=104, w=104, d_model=64)
        self.fusion2 = Fusion(img_size=52, patch_size=3, stride=2, in_chans=128, h=52, w=52, d_model=64)
        self.fusion3 = Fusion(img_size=26, patch_size=3, stride=2, in_chans=320, h=26, w=26, d_model=64)
        self.fusion4 = Fusion(img_size=13, patch_size=3, stride=1, in_chans=512, h=13, w=13, d_model=64)

        ###480*640_segformer
        # self.fusion1 = Fusion(img_size=(120, 160), patch_size=3, stride=2, in_chans=64, h=120, w=160, d_model=64)
        # self.fusion2 = Fusion(img_size=(60, 80), patch_size=3, stride=2, in_chans=128, h=60, w=80, d_model=64)
        # self.fusion3 = Fusion(img_size=(30, 40), patch_size=3, stride=2, in_chans=320, h=30, w=40, d_model=64)
        # self.fusion4 = Fusion(img_size=(15, 20), patch_size=3, stride=1, in_chans=512, h=15, w=20, d_model=64)

        # # 224*224
        # self.fusion1 = Fusion(img_size=56, patch_size=3, stride=2, in_chans=64, h=56, w=56)
        # self.fusion2 = Fusion(img_size=28, patch_size=3, stride=2, in_chans=128, h=28, w=28)
        # self.fusion3 = Fusion(img_size=14, patch_size=3, stride=2, in_chans=320, h=14, w=14)
        # self.fusion4 = Fusion(img_size=7, patch_size=3, stride=1, in_chans=512, h=7, w=7)

        self.bp1 = BoundaryPart()
        self.bp2 = BoundaryPart()
        self.bp3 = BoundaryPart()

        self.sp1 = SemanticPart()
        self.sp2 = SemanticPart()
        self.sp3 = SemanticPart()

        self.ep1 = EnhancePart()
        self.ep2 = EnhancePart()
        self.ep3 = EnhancePart()
 
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8x = nn.UpsamplingBilinear2d(scale_factor=8)

        self.cbl_out = nn.Sequential(
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=9, kernel_size=3, stride=1, padding=1),
                                   )
        self.cbl_b1 = ConvBNReLU(in_planes=64, out_planes=2, kernel_size=3)
        self.cbl_b2 = ConvBNReLU(in_planes=64, out_planes=2, kernel_size=3)
        self.cbl_b3 = ConvBNReLU(in_planes=64, out_planes=2, kernel_size=3)

        self.cbl_S1 = ConvBNReLU(in_planes=64, out_planes=2, kernel_size=3)
        self.cbl_S2 = ConvBNReLU(in_planes=64, out_planes=2, kernel_size=3)
        self.cbl_S3 = ConvBNReLU(in_planes=64, out_planes=2, kernel_size=3)

        self.cbl_d1 = ConvBNReLU(in_planes=64, out_planes=1, kernel_size=3)
        self.cbl_d2 = ConvBNReLU(in_planes=64, out_planes=1, kernel_size=3)

    def forward(self, rgb, t):
        out_r = self.rgb(rgb)
        out_t = self.t(t)
        # out_t1 = self.layer1_t(t)
        # out_t2 = self.layer2_t(out_t1)
        # out_t3 = self.layer3_t(out_t2)
        # out_t4 = self.layer4_t(out_t3)
        #
        # out_r1 = self.layer1_r(rgb)
        # out_r2 = self.layer2_r(out_r1)
        # out_r3 = self.layer3_r(out_r2)
        # out_r4 = self.layer4_r(out_r3)


        out_r1, out_r2, out_r3, out_r4 = out_r
        out_t1, out_t2, out_t3, out_t4 = out_t
        # print(out_r1.shape, out_r2.shape, out_r3.shape, out_r4.shape)
        # print(out_t1.shape, out_t2.shape, out_t3.shape, out_t4.shape)
        e1 = self.fusion1(out_r1, out_t1, f=None, NO=1)
        e2 = self.fusion2(out_r2, out_t2, f=e1, NO=2)
        e3 = self.fusion3(out_r3, out_t3, f=e2, NO=3)
        e4 = self.fusion4(out_r4, out_t4, f=e3, NO=4)

        ###layer1\2\3传给每一个单元
        b1 = self.bp1(NO=1, e1=e1, e2=e2, di=e4)
        s1 = self.sp1(NO=1, e3=e3, e4=e4, di=e4)
        d1 = self.ep1(NO=1, bi=b1, si=s1, dj=e4)

        b2 = self.bp2(NO=2, e1=e1, e2=e2, di=d1)
        s2 = self.sp2(NO=2, e3=e3, e4=e4, di=d1)
        d2 = self.ep2(NO=2, bi=b2, si=s2, dj=d1)

        b3 = self.bp3(NO=3, e1=e1, e2=e2, di=d2)
        s3 = self.sp3(NO=3, e3=e3, e4=e4, di=d2)
        d3 = self.ep3(NO=3, bi=b3, si=s3, dj=d2)

        # d3 = self.up2x(d3)
        # out = self.cbl_out(d3)
        b1_out = self.cbl_b1(self.up8x(b1))
        b2_out = self.cbl_b2(self.up4x(b2))
        b3_out = self.cbl_b3(self.up2x(b3))

        s1_out = self.cbl_S1(self.up8x(s1))
        s2_out = self.cbl_S2(self.up4x(s2))
        s3_out = self.cbl_S3(self.up2x(s3))

        # d1_out = self.cbl_d1(self.up8x(d1))
        # d2_out = self.cbl_d2(self.up4x(d2))
        out = self.cbl_out(self.up2x(d3))
       

        # return  b1_out, b2_out, b3_out, d1_out, d2_out, out
        return  b1_out, b2_out, b3_out, s1_out, s2_out,  s3_out, out


if __name__ == "__main__":
    a = torch.randn(2, 3, 480, 640)
    b = torch.randn(2, 3, 480, 640)
    model = nation()
    out = model(a, b)
    for i in range(len(out)):
        print(out[i].shape)
