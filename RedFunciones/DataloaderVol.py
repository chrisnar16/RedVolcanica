import json
import h5py

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms


class VolcanoDatasetH5(Dataset):

    def __init__(self, in_file, transform=None):
        # super(VolcanoDatasetH5, self).__init__()
        self.file = h5py.File(in_file, 'r')
        self.transform = transform

    def __getitem__(self, index):
        sampleB = self.file['X_train'][index, ...]
        tag = self.file['Y_train'][index, ...]
        # Preprocessing each image
        #tst = transforms.ToTensor()
        sample = torch.from_numpy(sampleB)
        #mean = sample.mean()
        #std = sample.std()
        maxx = sample.max()
        minn = sample.min()
        sample = ((sample - minn) /(maxx-minn))
        #sample = (2*(sample - minn) /(maxx-minn)) - 1
        #maxx = sample.max()
        sample = torch.unsqueeze(sample, 0)
        eps = 1e-7
        #sample = tst(sample)
        #sample = torch.log10(sample + eps)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, int(tag)

    def __len__(self):
        return self.file['X_train'].shape[0]


class VolcanoDataset(Dataset):

    def __init__(self, dir_name, json_name):
        # super(VolcanoDataset, self).__init__()
        self.dir_name = dir_name
        self.json_name = json_name
        with open(json_name, 'r') as fh:
            self.json = json.load(fh)

    def __len__(self):
        return len(self.json)

    def __getitem__(self, index):
        hashh = list(self.json)[index]
        tipo = list(self.json[hashh])[0]
        path = self.dir_name + hashh + '.npy'
        sample = torch.from_numpy(np.load(path))
        tag = -1
        if tipo == 'VT':
            tag = 0
        elif tipo == 'LP':
            tag = 1
        return sample, tag
