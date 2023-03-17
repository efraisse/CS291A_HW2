import os
import pandas as pd
from torchvision.io import read_image
from torchvision.datasets import CIFAR10
import pathlib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
import random

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
    
class PseudoOGMixDataset(Dataset):
    def __init__(self, transform=None):
        file_path_ti_500k = pathlib.Path(__file__).parent / "data" / "ti_500K_pseudo_labeled"
        data_dir='./data/'
        
        pi_500k_dataset = pd.read_pickle(file_path_ti_500k)
        
        # original CIFAR10 dataset and the semisup dataset
        # exclude the validation set info this time
        self.og_data = CIFAR10(data_dir, train=True).data[0:45000]
        self.og_labels = CIFAR10(data_dir, train=True).targets[0:45000]
        self.semisup_data = pi_500k_dataset["data"]
        self.semisup_labels = pi_500k_dataset["extrapolated_targets"]
        
        # shuffle the entire semisup dataset
        randomize = np.arange(len(self.semisup_data))
        np.random.shuffle(randomize)
        self.semisup_data = self.semisup_data[randomize]
        self.semisup_labels = self.semisup_labels[randomize]
        
        # grab length of CIFAR10 dataset of first elements
        sample_semisup_data = self.semisup_data[0:len(self.og_data)]
        sample_semisup_labels = self.semisup_labels[0:len(self.og_labels)]
    
        # combine the labels and images of both datasets
        self.transform = transform
        self.images = np.concatenate((sample_semisup_data, self.og_data), axis = 0)
        self.labels = np.concatenate((sample_semisup_labels, self.og_labels), axis = 0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx], 'RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
# if __name__ == "__main__":
#     file_path_ti_500k = pathlib.Path(__file__).parent / "data" / "ti_500K_pseudo_labeled"
#     pi_500k_dataset = pd.read_pickle(file_path_ti_500k)
    
#     print(pi_500k_dataset.keys())
#     data_dir='./data/'
    
#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])
    
#     val_ratio=0.1
    
#     train_size = int(50000 * (1 - val_ratio))
    
#     dataset = (pi_500k_dataset["data"], pi_500k_dataset["extrapolated_targets"])
    
#     random.shuffle(dataset)
#     print(dataset[0])
    
    