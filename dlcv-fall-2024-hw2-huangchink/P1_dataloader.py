import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms
import random

class MNISTMSVHN_Dataset(Dataset):
    def __init__(self, root_dir='./hw2_data/digits', dataset_type= 'mnistm', transform=None):
        """
        Args:
            root_dir (string): Directory with all the datasets.
            dataset_type (string): 'mnistm' or 'svhn' to specify the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.transform = transform
        
        # Load the corresponding CSV
        csv_file = os.path.join(root_dir, dataset_type, 'train.csv')
        self.data_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.dataset_type, 'data', self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # Get label from CSV file
        label = int(self.data_frame.iloc[idx, 1])
        digit_cond = torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()
        if self.transform:
            image = self.transform(image)
        
        # Generate conditional label for the dataset type: 0 for MNIST-M, 1 for SVHN
        dataset_label = 0 if self.dataset_type == 'mnistm' else 1
        dataset_cond = torch.nn.functional.one_hot(torch.tensor(dataset_label), num_classes=2).float()

        return image, digit_cond, dataset_cond

def get_dataloader(batch_size=64, root_dir='./hw2_data/digits', transform=None):
    # Define transformations
    if transform is None:
        transform = transforms.Compose([
            # transforms.Resize((32, 32)),  # Resize images to 32x32 for compatibility
            transforms.ToTensor(),
        ])
    
    # Create dataset objects for both MNIST-M and SVHN
    mnistm_dataset = MNISTMSVHN_Dataset(root_dir=root_dir, dataset_type='mnistm', transform=transform)
    svhn_dataset = MNISTMSVHN_Dataset(root_dir=root_dir, dataset_type='svhn', transform=transform)
    print(f'mnistm_dataset:{len(mnistm_dataset)}')
    print(f'svhn_dataset:{len(svhn_dataset)}')
    # # 隨機選擇一個索引
    # random_idx = random.randint(0, len(mnistm_dataset) - 1)

    # # 獲取對應的圖片、標籤和資料集標籤
    # image, label, dataset_label = mnistm_dataset[random_idx]

    # print(f"Random index: {random_idx}")
    # print(f"Image shape: {image.size}")  # PIL image size
    # print(f"Label: {label}")
    # print(f"Dataset label: {dataset_label}")

    # Combine the datasets using ConcatDataset
    combined_dataset = torch.utils.data.ConcatDataset([mnistm_dataset, svhn_dataset])
    
    # Create a DataLoader
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataloader

# Test the DataLoader
if __name__ == '__main__':
    dataloader = get_dataloader(batch_size=64, root_dir='./hw2_data/digits')
    
    for images, labels, dataset_labels in dataloader:
        print(f"Image batch shape: {images.size()}")
        print(f"Label batch shape: {labels.size()}")
        print(f"Dataset labels batch shape: {dataset_labels.size()}")
        break
