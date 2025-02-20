import torch
from torch.utils.data import Dataset
import numpy as np
import os

class PoseDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pose_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.npy'):
                    self.pose_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.pose_files)

    def __getitem__(self, idx):
        pose_path = self.pose_files[idx]
        pose_data = np.load(pose_path)
        return torch.from_numpy(pose_data)
