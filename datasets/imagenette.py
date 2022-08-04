import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('imagenette')
class Imagenette(Dataset):

    def __init__(self, root_path, split, augment):
        root_path = os.path.join(root_path, split)
        classes = sorted(os.listdir(root_path))
        self.data = []
        for c in classes:
            filenames = sorted(os.listdir(os.path.join(root_path, c)))
            for f in filenames:
                self.data.append(os.path.join(root_path, c, f))
        if augment == 'none':
            self.transform = transforms.Compose([])
        elif augment == 'random_crop_178':
            self.transform = transforms.Compose([
                transforms.RandomCrop(178),
                transforms.RandomHorizontalFlip(),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.data[idx]).convert('RGB'))
