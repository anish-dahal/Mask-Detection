import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torch.utils.data import (
    random_split,
    DataLoader,
    Dataset
)

torch.manual_seed(100)

class MaskDetectionDataSet(Dataset):
    def __init__(self, data_dir):
        self.dataset = ImageFolder(
            data_dir,
            tt.Compose([
            tt.Resize(size=(32,32)),
            tt.ToTensor(),
            tt.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index] 


def dataset_split(dataset, val_ratio = 0.2):
    train_ds, valid_ds = random_split(dataset, [1-val_ratio, val_ratio])
    return train_ds, valid_ds
    

def dataloader(dataset, batch_size=16, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)