import json
import os
import glob
import pickle

import torch.utils.data as data
import pretrainedmodels.utils as utils

from PIL import Image

class PairData(data.Dataset):
    def __init__(self, phase='train',
                 transform=None):
        super(PairData, self).__init__()
        assert phase in ('train', 'val')
        self.phase = phase
        self.transform = transform
        self.data_dir = '../Synthdata'
        
        if phase == 'train':
            self.path = os.path.join(self.data_dir, 'train_pair.txt')
        else:
            self.path = os.path.join(self.data_dir, 'test_pair.txt')

        with open(self.path) as fid:
            self.Sets = [line.strip() for line in fid.readlines()]

    def __len__(self):
        return len(self.Sets)

    def __getitem__(self, index):
        infos = self.Sets[index].split(',')
        image1 = Image.open(os.path.join(self.data_dir, infos[0])).convert('RGB')
        image2 = Image.open(os.path.join(self.data_dir, infos[1])).convert('RGB')
        label = int(infos[2])

        if self.transform:
            image1, image2 = self.transform(image1, image2)

        return image1, image2, label


def makeloader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory)