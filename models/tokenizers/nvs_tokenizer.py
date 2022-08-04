import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from models import register
from utils import poses_to_rays


@register('nvs_tokenizer')
class NvsTokenizer(nn.Module):

    def __init__(self, input_size, patch_size, dim, padding=0, img_channels=3):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.patch_size = patch_size
        self.padding = padding
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * (img_channels + 3 + 3), dim)
        self.grid_shape = ((input_size[0] + padding[0] * 2) // patch_size[0],
                           (input_size[1] + padding[1] * 2) // patch_size[1])

    def forward(self, data):
        imgs = data['support_imgs']
        B = imgs.shape[0]
        H, W = imgs.shape[-2:]
        rays_o, rays_d = poses_to_rays(data['support_poses'], H, W, data['support_focals'])
        rays_o = einops.rearrange(rays_o, 'b n h w c -> b n c h w')
        rays_d = einops.rearrange(rays_d, 'b n h w c -> b n c h w')

        x = torch.cat([imgs, rays_o, rays_d], dim=2)
        x = einops.rearrange(x, 'b n d h w -> (b n) d h w')
        p = self.patch_size
        x = F.unfold(x, p, stride=p, padding=self.padding)
        x = einops.rearrange(x, '(b n) ppd l -> b (n l) ppd', b=B)

        x = self.prefc(x)
        return x
