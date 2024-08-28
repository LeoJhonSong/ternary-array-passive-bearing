import os
from typing import Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class Array_Data_DataSet(Dataset):
    """
    Returns
    -------
    data_n : torch.Tensor[torch.float32]
        若label_type=='direction', shape: `[channels, t_len]`,
        若label_type=='position', shape: `(sample_intervals, channels, t_len)`
    fs : torch.int64
    label_n : torch.Tensor[torch.float32]
        若label_type=='direction', shape: `[2]` or `[3]`, 暂时只从`[sample_intervals]`取第一个时间段,
        若label_type=='position', shape: `(sample_intervals, 2)` or `(sample_intervals, 3)`
    """

    def __init__(self, folder_path: str, seq: bool, label_type: Literal['direction', 'position'], distance_range: Tuple[float | int, float | int] = (800, 1000)):
        # folder_path下所有文件的列表
        self.folder_path = folder_path
        self.filenames = [f'{folder_path}/{filename}' for filename in os.listdir(folder_path)]
        self.seq = seq
        self.label_type = label_type
        self.distance_range = distance_range

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        dataDict = np.load(self.filenames[idx])
        data_n, _, r_n, angle_n, t_bound_n = dataDict['data_segments'], dataDict['fs'], dataDict['r_n'], dataDict['angle_n'], dataDict['t_bound_n']
        data_n = torch.tensor(data_n, dtype=torch.float32)  # shape: (sample_intervals, channels, t_len)
        r_n = torch.tensor(r_n, dtype=torch.float32).reshape(-1)
        angle_n = torch.tensor(angle_n, dtype=torch.float32).reshape(-1)
        theta_n = torch.deg2rad(angle_n)
        t_bound_n = torch.tensor(t_bound_n, dtype=torch.float32).reshape(-1, 2)
        if self.label_type == 'direction':
            label_n = torch.stack((
                torch.cos(theta_n),
                torch.sin(theta_n),
            ), dim=1)
        elif self.label_type == 'position':
            label_n = torch.stack((
                torch.cos(theta_n),
                torch.sin(theta_n),
                self.map_distance(r_n),
            ), dim=1)
        else:
            raise ValueError('Wrong label_type')
        if self.seq:
            return data_n, label_n
        else:
            return data_n[0].unsqueeze(0), label_n[0]

    def map_distance(self, distances: torch.Tensor):
        """
        Parameters
        ----------
        distances : torch.Tensor
            shape: [batch]

        Returns
        -------
        normalized_distances : torch.Tensor
            shape: [batch]
        """
        # limit to known range, maybe out of range because of precision
        distances = distances.clamp(min=self.distance_range[0], max=self.distance_range[1])
        # map from distance_range to [0, 1]
        distances = (distances - self.distance_range[0]) / (self.distance_range[1] - self.distance_range[0])
        return distances

    def labels2angles(self, labels: torch.Tensor):
        """
        Parameters
        ----------
        labels : torch.Tensor
            shape: [batch, 2 or 3] or [batch, seq_len, 2 or 3]
        Returns
        -------
        angles : torch.Tensor
            shape: [batch] or [batch, seq_len]
        """
        return torch.rad2deg(torch.atan2(labels[..., 1], labels[..., 0]))

    def labels2positions(self, labels: torch.Tensor):
        """
        Parameters
        ----------
        labels : torch.Tensor
            shape: [batch, 3] or [batch, seq_len, 3]
        Returns
        -------
        distances : torch.Tensor
            shape: [batch] or [batch, seq_len]
        angles : torch.Tensor
            shape: [batch] or [batch, seq_len]
        """
        # limit to known range, maybe out of range because of precision
        distances = labels[..., 2].clamp(min=0, max=1)
        # map from [0, 1] to distance_range
        distances = distances * (self.distance_range[1] - self.distance_range[0]) + self.distance_range[0]
        return distances, self.labels2angles(labels)


class Curriculum_Array_Dataset(Array_Data_DataSet):
    def __init__(self, folder_path: str, seq: bool, curriculum: Tuple[int, int, int, int], label_type: Literal['direction', 'position'], distance_range: Tuple[float | int, float | int] = (800, 1000)):
        super().__init__(folder_path, seq, label_type, distance_range)
        self.subfolders = [f'{folder_path}/{subfolder}' for subfolder in ['ideal', '30', '20', '10', '5']]
        self.subfolders = [subfolder for subfolder in self.subfolders if os.path.exists(subfolder)]
        self.file_tree = [[f'{subfolder}/{filename}' for filename in os.listdir(subfolder)] for subfolder in self.subfolders]
        self.curriculum = curriculum
        self.step(0)

    def step(self, epoch: int):
        for i in range(len(self.curriculum)):
            if epoch <= self.curriculum[i]:
                self.filenames = [filename for filenames in self.file_tree[:i + 1] for filename in filenames]
                break
        else:
            self.filenames = [filename for filenames in self.file_tree[:i + 2] for filename in filenames]
