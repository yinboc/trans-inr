import os
from PIL import Image

from torch.utils.data import Dataset

from datasets import register


@register('celeba')
class Celeba(Dataset):

    def __init__(self, root_path, split):
        if split == 'train':
            s, t = 1, 162770
        elif split == 'val':
            s, t = 162771, 182637
        elif split == 'test':
            s, t = 182638, 202599
        self.data = []
        for i in range(s, t + 1):
            path = os.path.join(root_path, 'img_align_celeba', 'img_align_celeba', f'{i:06}.jpg')
            self.data.append(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return Image.open(self.data[idx])
