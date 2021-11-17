import json

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class VolcanoDataset(Dataset):

    def __init__(self, dir_name, json_name):
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



