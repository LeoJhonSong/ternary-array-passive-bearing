import os

import numpy as np
import torch
from torch.utils.data import Dataset


class Array_Data_DataSet(Dataset):
    def __init__(self, folder_path):
        # folder_path下所有文件的列表
        self.folder_path = folder_path
        self.filenames = os.listdir(folder_path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        dataDict = np.load(f'{self.folder_path}/{self.filenames[idx]}')
        data_n, fs, r_n, angle_n = dataDict['data_segments'], dataDict['fs'], dataDict['r_n'], dataDict['angle_n']
        data_n = torch.tensor(data_n, dtype=torch.float32)  # shape: (sample_intervals, channels t_len)
        r_n = torch.tensor(r_n, dtype=torch.float32).reshape(1, -1)
        angle_n = torch.tensor(angle_n, dtype=torch.float32).reshape(1, -1)
        return data_n, fs, r_n, angle_n, self.filenames[idx]
