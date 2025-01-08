import os
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 定義資料的 transforms
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自訂的 Dataset 類別
class MiniImageNetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # 確保圖片是 RGB 格式
        if self.transform:
            image = self.transform(image)
        return image
class OfficeDataset(Dataset):
    def __init__(self, csv_file, img_dir, mode='train', transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with filenames and (optionally) labels.
            img_dir (str): Directory with all the images.
            mode (str): 'train' or 'val'. In 'train' mode, labels are available; in 'val' mode, only filenames.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode

        # 讀取 CSV 檔案
        self.data = pd.read_csv(csv_file)
        if mode == 'train':
            # 如果是 train 模式，會有 label
            self.filenames = self.data['filename'].values
            self.labels = self.data['label'].values
        else:
            # 如果是 val 模式，只有 filename
            self.filenames = self.data['filename'].values
            self.labels = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            label = self.labels[idx]
            return image, label  # 回傳圖片和標籤
        else:
            filename = self.filenames[idx]
            return image, filename  # val 模式下回傳圖片和文件名



# 測試 dataloader 是否正常工作
if __name__ == "__main__":
    # 資料夾路徑
    # data_dir = './hw1_data/p1_data/mini/train'

    # # 創建 Dataset 和 DataLoader
    # dataset = MiniImageNetDataset(data_dir, transform=TRANSFORM_IMG)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    # print(len(dataset))
    # print(len(dataloader))

    # mode = 'train'  # 或者 'val'

    # if mode == 'train':
    #     csv_file = './hw1_data/p1_data/office/train.csv'
    #     img_dir = './hw1_data/p1_data/office/train/'
    # else:
    #     csv_file = './hw1_data/p1_data/office/val.csv'
    #     img_dir = './hw1_data/p1_data/office/val/'

    # # 創建 Dataset 和 DataLoader
    # dataset = OfficeDataset(csv_file, img_dir, mode=mode, transform=TRANSFORM_IMG)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # print(f"Dataset size: {len(dataset)}")
    # for i, data in enumerate(dataloader):
    #     if mode == 'train':
    #         images, labels = data
    #         print(f"Batch {i+1} - Images: {images.shape}, Labels: {labels.shape}")
    #     else:
    #         images = data
    #         print(f"Batch {i+1} - Images: {images.shape}")
    mode = 'val'  # 可以切換成 'train' 測試不同模式
    print('reading files')
    if mode == 'train':
        csv_file = './hw1_data/p1_data/office/train.csv'
        img_dir = './hw1_data/p1_data/office/train/'
    else:
        csv_file = './hw1_data/p1_data/office/val.csv'
        img_dir = './hw1_data/p1_data/office/val/'

    # 創建 Dataset 和 DataLoader
    dataset = OfficeDataset(csv_file, img_dir, mode=mode, transform=TRANSFORM_IMG)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    print(f"Dataset size: {len(dataset)}")
    for i, data in enumerate(dataloader):
        if mode == 'train':
            images, labels = data
            print(f"Batch {i+1} - Images: {images.shape}, Labels: {labels.shape}")
        else:
            images, filenames = data
            print(f"Batch {i+1} - Images: {images.shape}, Filenames: {filenames[0]}")