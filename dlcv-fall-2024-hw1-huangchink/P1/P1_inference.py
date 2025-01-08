import torch
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms
import csv
from P1.P1_dataloader import OfficeDataset
from torchvision import models
from torch import nn

class ClassificationModel(nn.Module):
    def __init__(self, num_classes=65):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.backbone, self.num_ftrs = self.load_model()

        self.classifier = nn.Sequential(
            nn.Linear(self.num_ftrs, self.num_classes)
        )

    def load_model(self):
        resnet = models.resnet50(weights=None)     
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Identity()
        return resnet, num_ftrs

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        out = self.classifier(features)
        return out
    
parser = argparse.ArgumentParser()
parser.add_argument("--image_csv", type=str)
parser.add_argument("--image_folder", type=str)
parser.add_argument("--output_csv", type=str)
args = parser.parse_args()

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = OfficeDataset(csv_file=args.image_csv,
                            img_dir=args.image_folder, mode='val', transform=transform_test)
batch_size = 128
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassificationModel().to(device)
checkpoint_path = './P1/checkpoint/C_best_model.pth'
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict)
model.eval()

results_dict = {}  # 創建一個字典來存儲結果

with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        input_images, filename = data
        input_images = input_images.to(device)
        outputs = model(input_images)
        
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()

        # 將每個結果添加到字典中
        for i in range(len(predicted)):  
            image_id = batch_idx * batch_size + i
            results_dict[image_id] = {'filename': filename[i], 'label': predicted[i]}

# 所有 batch 完成後，一次性寫入到 CSV
with open(args.output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'filename', 'label'])
    
    # 從字典中提取結果並寫入 CSV
    for image_id, result in results_dict.items():
        writer.writerow([image_id, result['filename'], result['label']])

print(f"Results saved to {args.output_csv}")
