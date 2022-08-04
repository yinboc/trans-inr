# Reference: https://github.com/tancik/learnit/blob/main/Experiments/shapenet.ipynb
import os
import json

import imageio
import numpy as np
import torch
import einops
from torch.utils.data import Dataset

from datasets import register


@register('learnit_shapenet')
class LearnitShapenet(Dataset):

    def __init__(self, root_path, category, split, n_support, n_query, views_rng=None, repeat=1):
        with open(os.path.join(root_path, category[:-len('s')] + '_splits.json'), 'r') as f:
            obj_ids = json.load(f)[split]
        _data = [os.path.join(root_path, category, _) for _ in obj_ids]
        self.data = []
        for x in _data:
            if os.path.exists(os.path.join(x, 'transforms.json')):
                self.data.append(x)
            else:
                print(f'Missing obj at {x}, skipped.')
        self.n_support = n_support
        self.n_query = n_query
        self.views_rng = views_rng
        self.repeat = repeat

    def __len__(self):
        return len(self.data) * self.repeat

    def __getitem__(self, idx):
        idx %= len(self.data)
        
        train_ex_dir = self.data[idx]
        with open(os.path.join(train_ex_dir, 'transforms.json'), 'r') as fp:
            meta = json.load(fp)
        camera_angle_x = float(meta['camera_angle_x'])
        frames = meta['frames']
        if self.views_rng is not None:
            frames = frames[self.views_rng[0]: self.views_rng[1]]

        frames = np.random.choice(frames, self.n_support + self.n_query, replace=False)

        imgs = []
        poses = []
        for frame in frames:
            fname = os.path.join(train_ex_dir, os.path.basename(frame['file_path']) + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        H, W = imgs[0].shape[:2]
        assert H == W
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        imgs = imgs[..., :3] * imgs[..., -1:] + (1 - imgs[..., -1:])
        poses = np.array(poses).astype(np.float32)

        imgs = einops.rearrange(torch.from_numpy(imgs), 'n h w c -> n c h w')
        poses = torch.from_numpy(poses)[:, :3, :4]
        focal = torch.ones(len(poses), 2) * float(focal)
        t = self.n_support
        return {
            'support_imgs': imgs[:t],
            'support_poses': poses[:t],
            'support_focals': focal[:t],
            'query_imgs': imgs[t:],
            'query_poses': poses[t:],
            'query_focals': focal[t:],
            'near': 2,
            'far': 6,
        }
