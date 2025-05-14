import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class PostureDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.labels = df["label"]
        self.data = df.drop("label", axis=1).values.astype("float32")

        self.le = LabelEncoder()
        self.label_ids = torch.tensor(self.le.fit_transform(self.labels), dtype=torch.long)

        # reshape: (samples, 3 frames, 21 features)
        self.data = torch.tensor(self.data, dtype=torch.float32).reshape(-1, 3, 21)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label_ids[idx]

    def get_label_encoder(self):
        return self.le
