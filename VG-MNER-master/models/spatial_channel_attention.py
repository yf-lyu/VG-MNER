import torch.nn as nn
import torch
from einops.einops import rearrange


class ParamDecoder(nn.Module):
    def __init__(self, mu_dim, need_in_dim, need_out_dim, k=30):
        super(ParamDecoder, self).__init__()
        self.need_in_dim = need_in_dim
        self.need_out_dim = need_out_dim
        self.k = k
        self.decoder = nn.Linear(mu_dim, need_in_dim * k) 
        self.V = nn.parameter.Parameter(torch.zeros(k, need_out_dim)) # S

    def forward(self, t_feat):
        B = t_feat.shape[0]
        U = self.decoder(t_feat).reshape(B, self.need_in_dim, self.k)  # B x need_in_dim x k
        param=torch.einsum('bik,kj->bij', U , self.V).reshape(B, -1)
        return param


class DynamicLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, mu_dim: int, bias=True, k=30):
        super(DynamicLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mu_dim = mu_dim
        self.bias=bias
        self.decoder = ParamDecoder(mu_dim, in_dim + 1, out_dim, k=k)

    def forward(self, x, mu):
        param=rearrange(self.decoder(mu), 'B (dim_A dim_B) -> B dim_A dim_B', dim_A=self.in_dim+1, dim_B=self.out_dim)
        weight=param[:,:-1,:]
        bias=param[:, -1, :]
        x=torch.einsum('bd...,bde->be...', x, weight)
        if self.bias:
            bias=bias.view(((bias.shape[0],)+(bias.shape[-1],)+(1,)*(len(x.size())-2)))
            x=x+bias
        return x


class SpatialAttention(nn.Module):
    def __init__(self, in_planes=1024, text_dim=768, k=30):
        '''空间注意力'''
        super(SpatialAttention, self).__init__()
        self.spatial_dynlinear = DynamicLinear(in_dim=in_planes, out_dim=1, mu_dim=text_dim, k=k)
        self.conv = nn.Conv2d(1, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, text_feat):   # x 的输入格式是：[batch_size, C, H, W]
        x = self.spatial_dynlinear(x, text_feat)
        x = self.conv(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, text_dim=768, k=30):
        '''通道注意力'''
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.max_dynlinear = DynamicLinear(in_dim=in_planes, out_dim=in_planes, mu_dim=text_dim, k=k)
        self.avg_dynlinear = DynamicLinear(in_dim=in_planes, out_dim=in_planes, mu_dim=text_dim, k=k)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, text_feat):   # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        avg_dyn = self.avg_dynlinear(avg_out, text_feat)

        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        max_dyn = self.max_dynlinear(max_out, text_feat)

        out = avg_dyn + max_dyn
        return self.sigmoid(out)
