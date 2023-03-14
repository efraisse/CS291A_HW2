import os
import pandas as pd
from torchvision.io import read_image
import pathlib
from torch.utils.data import Dataset
from PIL import Image

class PseudoLabelDataset(Dataset):
    def __init__(self, transform=None):
        file_path_ti_500k = pathlib.Path(__file__).parent / "data" / "ti_500K_pseudo_labeled"
        pi_500k_dataset = pd.read_pickle(file_path_ti_500k)
    
        self.transform = transform
        self.images = pi_500k_dataset["data"]
        self.labels = pi_500k_dataset["extrapolated_targets"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx], 'RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label