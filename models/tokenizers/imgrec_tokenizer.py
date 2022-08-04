import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register


@register('imgrec_tokenizer')
class ImgrecTokenizer(nn.Module):

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
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * img_channels, dim)
        n_patches = ((input_size[0] + padding[0] * 2) // patch_size[0]) * ((input_size[1] + padding[1] * 2)  // patch_size[1])
        self.posemb = nn.Parameter(torch.randn(n_patches, dim))

    def forward(self, data):
        x = data['inp']
        p = self.patch_size
        x = F.unfold(x, p, stride=p, padding=self.padding) # (B, C * p * p, L)
        x = x.permute(0, 2, 1).contiguous()
        x = self.prefc(x) + self.posemb.unsqueeze(0)
        return x
