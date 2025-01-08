import torch
import torch.nn as nn
import torchvision.models as models

class DeepLabV3Model(nn.Module):
    def __init__(self, model_type='resnet101', num_classes=7):
        """
        Args:
            model_type (str): 指定要使用的模型，'resnet101' 或 'mobilenet_v3_large'
            num_classes (int): 最終輸出的類別數，預設為 7
            pretrained (bool): 是否加載預訓練權重
        """
        super(DeepLabV3Model, self).__init__()

        if model_type == 'resnet101':
            print("Loading DeepLabV3 with ResNet101 backbone")
            self.model = models.segmentation.deeplabv3_resnet101(weight = models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,
                                                            num_classes = 7,
                                                            aux_loss = True)



        elif model_type == 'mobilenet_v3_large':
            print("Loading DeepLabV3 with MobileNetV3-Large backbone")
            self.model =models.segmentation.deeplabv3_mobilenet_v3_large(weight = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
                                                    num_classes = num_classes,
                                                    aux_loss = True)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'resnet101' or 'mobilenet_v3_large'.")
        # self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

    def forward(self, x):

        return self.model(x)  

# 測試代碼
if __name__ == "__main__":
    # 測試加載不同的模型
    for model_type in ['resnet50', 'mobilenet_v3_large']:
        print(f"\nTesting model type: {model_type}")
        model = DeepLabV3Model(model_type=model_type, num_classes=7)
        sample_input = torch.randn(10, 3, 512, 512)  # 測試輸入
        output = model(sample_input)
        pred = output['out'].max(1, keepdim=False)[1].cpu().numpy()
        print(pred.shape)