import torch
import torch.nn as nn
import torchvision.models as models



class FCN32s(nn.Module):
    def __init__(self, n_class=7):
        super(FCN32s, self).__init__()
        self.vgg16 = models.vgg16(weights='VGG16_Weights.IMAGENET1K_FEATURES')
        self.features_extractor = self.vgg16.features
        self.FCN32 = nn.Sequential(
            nn.Conv2d(512, 4096, 7,padding=3),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),

            nn.Conv2d(4096, 4096, 1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),


            nn.Conv2d(4096, n_class,1),
            nn.BatchNorm2d(n_class),

            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=64, padding=16,stride=32, bias=False)

        )




    def forward(self, x):
        x = self.features_extractor(x)
        x = self.FCN32(x)


        return x
if __name__ == "__main__":
    # 定義輸入的 batch size, 通道數, 高度, 寬度
    input_tensor = torch.randn(1, 3, 512, 512)  # 假設輸入圖像大小為 224x224，RGB 3 通道

    # 創建 FCN32s 模型
    model = FCN32s(n_class=7)

    # 將模型設置為評估模式 (如果是測試推理的話)
    model.eval()

    # 前向傳播，獲得輸出
    with torch.no_grad():  # 關閉梯度計算
        output = model(input_tensor)
    
    # 打印輸出尺寸
    print(f"輸出尺寸: {output.size()}")
