import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PostureDataset(Dataset):
    def __init__(self, csv_file, mode='train', test_size=0.2, val_size=0.1):
        self.data = pd.read_csv(csv_file)

        self.labels = sorted(self.data['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        X = self.data.drop(columns=['label']).values.astype(np.float32)
        y = self.data['label'].map(self.label_to_idx).values.astype(np.int64)

        # Train/Val/Test 분할
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_size, stratify=y_trainval, random_state=42)

        if mode == 'train':
            self.X, self.y = X_train, y_train
        elif mode == 'val':
            self.X, self.y = X_val, y_val
        elif mode == 'test':
            self.X, self.y = X_test, y_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
