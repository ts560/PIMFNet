import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch
import os
class CustomResNet50(nn.Module):
    def __init__(self, input_channels=1, num_classes=1000):
        super(CustomResNet50, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model_feat = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.model_layer=nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )
        self.model_fc1 = nn.Linear(in_features=resnet.fc.in_features, out_features=256)
        self.model_fc2 = nn.Linear(in_features=256, out_features=num_classes)
    def forward(self, x):
        y = self.model_feat(x)
        y = self.model_layer(y)
        y_shared = self.model_fc1(y)
        y_distinct=self.model_fc1(y)
        y_pred = self.model_fc2(y_distinct)
        return y_shared, y_distinct, y_pred

class PhyModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=1000):
        super(PhyModel, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model_feat = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.model_layer=nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )
        self.model_fc1 = nn.Linear(in_features=30, out_features=256)
        self.model_fc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.model_fc3 = nn.Linear(in_features=resnet.fc.in_features, out_features=30)


    def forward(self, x):
        inputs1 = torch.fft.fft(torch.complex(x, torch.zeros_like(x))).abs()
        positions = [17, 35, 52, 70, 86, 104, 121, 138, 156, 173, 190, 208, 259, 517, 778, 1046, 1297, 1555, 1809, 2064,
                     2324, 242, 500, 761, 1029, 1280, 1538, 1792, 2047, 2307]  # S2
        extracted_values = inputs1[:, :, positions, :]
        extracted_values = extracted_values.view(x.size(0), 30)
        y = self.model_feat(x)
        y = self.model_layer(y)
        y_30 = self.model_fc3(y)
        fused_30=extracted_values+y_30
        y_256=self.model_fc1(fused_30)
        y_pred = self.model_fc2(y_256)
        return fused_30,y_256,y_pred,extracted_values

