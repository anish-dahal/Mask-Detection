import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torch.utils.data import (
    random_split,
    DataLoader,
    Dataset
)

torch.manual_seed(100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MaskDetectionDataSet(Dataset):
    def __init__(self, data_dir):
        self.dataset = ImageFolder(
            data_dir,
            tt.Compose([
            tt.Resize(size=(256, 256)),
            tt.CenterCrop(224),
            tt.ToTensor(),
            tt.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        )
        self.classes = self.dataset.classes
    
    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, index):
        return self.dataset[index][0].to(device), torch.tensor(self.dataset[index][1]).to(device)


def dataset_split(dataset, val_ratio = 0.2):
    train_ds, valid_ds = random_split(dataset, [1-val_ratio, val_ratio])
    return train_ds, valid_ds
    

def dataloader(dataset, batch_size=16, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)