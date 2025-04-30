import torch
from torch.utils.data import Dataset
import pandas as pd

class PostureDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label_map = {
            "good_posture": 0,
            "shoulder_tilt": 1,
            "forward_head": 2,
            "head_tilt": 3,
            "leaning_back": 4
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = self.label_map[row['label']]
        coords = row.iloc[1:].values.astype('float32')
        return torch.tensor(coords), label
