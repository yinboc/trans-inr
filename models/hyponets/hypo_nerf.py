import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import register
from .layers import batched_linear_mm


@register('hypo_nerf')
class HypoNerf(nn.Module):

    def __init__(self, use_viewdirs=False, depth=6, hidden_dim=256, use_pe=True, pe_dim=40):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.depth = depth
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.param_shapes = dict()

        in_dim = 3
        if use_pe:
            last_dim = in_dim * pe_dim
        else:
            last_dim = in_dim
        for i in range(depth - 1):
            self.param_shapes[f'wb{i}'] = (last_dim + 1, hidden_dim)
            last_dim = hidden_dim

        if self.use_viewdirs:
            self.param_shapes['viewdirs_fc'] = (3 + 1, hidden_dim // 2)
            self.param_shapes['density_fc'] = (hidden_dim + 1, 1)
            self.param_shapes['rgb_fc1'] = (hidden_dim + hidden_dim // 2 + 1, hidden_dim)
            self.param_shapes['rgb_fc2'] = (hidden_dim + 1, 3)
        else:
            self.param_shapes['rgb_density_fc'] = (hidden_dim + 1, 4)

        self.relu = nn.ReLU()
        self.params = None

    def set_params(self, params):
        self.params = params

    def convert_posenc(self, x):
        w = 2**torch.linspace(0, 8, self.pe_dim // 2, device=x.device)
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return x

    def forward(self, x, viewdirs=None):
        B, query_shape = x.shape[0], x.shape[1: -1]

        x = x.view(B, -1, 3)
        if self.use_pe:
            x = self.convert_posenc(x)

        if self.use_viewdirs:
            viewdirs = viewdirs.contiguous().view(B, -1, 3)
            viewdirs = F.normalize(viewdirs, dim=-1)
            viewdirs = batched_linear_mm(viewdirs, self.params['viewdirs_fc'])
            viewdirs = self.relu(viewdirs)

        for i in range(self.depth - 1):
            x = batched_linear_mm(x, self.params[f'wb{i}'])
            x = self.relu(x)

        if self.use_viewdirs:
            density = batched_linear_mm(x, self.params['density_fc'])
            x = torch.cat([x, viewdirs], dim=-1)
            x = batched_linear_mm(x, self.params['rgb_fc1'])
            x = self.relu(x)
            rgb = batched_linear_mm(x, self.params['rgb_fc2'])
            out = torch.cat([rgb, density], dim=-1)
        else:
            out = batched_linear_mm(x, self.params['rgb_density_fc'])

        return out.view(B, *query_shape, -1)
