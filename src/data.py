"""Dataset utilities. Initial scaffold."""
import os
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = root
        self.split = split

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError
