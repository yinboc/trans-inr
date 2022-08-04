from torch.utils.data import Dataset
from torchvision import transforms

import datasets
from datasets import register


@register('imgrec_dataset')
class ImgrecDataset(Dataset):

    def __init__(self, imageset, resize):
        self.imageset = datasets.make(imageset)
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.imageset)

    def __getitem__(self, idx):
        x = self.transform(self.imageset[idx])
        return {'inp': x, 'gt': x}
