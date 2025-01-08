import torch
from torch import nn
from torchvision import models

class ClassificationModel(nn.Module):
    def __init__(self, setting='C', num_classes=65):
        """
        Args:
            setting (str): 設定 A, B, C, D, E 來加載對應的模型.
            num_classes (int): 類別數，默認為 65.
        """
        super(ClassificationModel, self).__init__()
        self.setting = setting
        self.num_classes = num_classes
        self.backbone ,self.num_ftrs=  self.load_model()

        # 獲取 ResNet 最後一層全連接層的輸入特徵數

        # 自定義的分類器
        self.classifier = nn.Sequential(
            nn.Linear(self.num_ftrs, self.num_classes)
        )

    def load_model(self):
        """
        根據 setting 加載不同的預訓練模型，並去掉最後的全連接層以適應 65 個類別
        """
        resnet = models.resnet50(weights=None)

        if self.setting == 'A':
            print("Setting A: Initializing ResNet50 without pretraining.")
        
        elif self.setting == 'B':
            print("Setting B: Loading TAs' pre-trained model.")
            checkpoint_path = '../hw1_data/p1_data/pretrain_model_SL.pt'
            state_dict = torch.load(checkpoint_path)
            resnet.load_state_dict(state_dict)
        
        elif self.setting == 'C':
            print("Setting C: Loading Your SSL pre-trained backbone.")
            checkpoint_path = './checkpoint/SSL-Resnet.pt'
            state_dict = torch.load(checkpoint_path)
            resnet.load_state_dict(state_dict)

        elif self.setting == 'D':
            print("Setting D: Loading TAs' pre-trained model and freezing backbone.")
            checkpoint_path = '../hw1_data/p1_data/pretrain_model_SL.pt'
            state_dict = torch.load(checkpoint_path)
            resnet.load_state_dict(state_dict)
            for param in resnet.parameters():
                param.requires_grad = False  # 凍結 backbone
            print("Backbone frozen.")

        elif self.setting == 'E':
            print("Setting E: Loading Your SSL pre-trained backbone and freezing backbone.")
            checkpoint_path = './checkpoint/SSL-Resnet.pt'
            state_dict = torch.load(checkpoint_path)
            resnet.load_state_dict(state_dict)
            for param in resnet.parameters():
                param.requires_grad = False  # 凍結 backbone
            print("Backbone frozen.")
        
        else:
            raise ValueError(f"Unknown setting: {self.setting}")
        
        # 去掉 ResNet50 的最後一層全連接層
        num_ftrs = resnet.fc.in_features

        resnet.fc = nn.Identity()

        return resnet,num_ftrs

    def forward(self, x):
        # 通過 backbone 提取特徵
        features = self.backbone(x)
        
        # 展平特徵以傳入 classifier
        features = torch.flatten(features, 1)
        
        # 通過分類器進行分類
        out = self.classifier(features)
        return out

# 測試代碼
if __name__ == "__main__":
    # 測試載入不同的模型
    for setting in ['A', 'B', 'C', 'D', 'E']:
        print(f"\nTesting setting {setting}:")
        classification_model = ClassificationModel(setting=setting)
        model = classification_model
        print(f"Model for setting {setting} loaded successfully.")
