import torch.nn as nn
import torch

class CNN_1D_P1(nn.Module):
    def __init__(self, input_channels=1, num_classes=9):
        super(CNN_1D_P1, self).__init__()
        self.model_feat=nn.Sequential(
            nn.Conv1d(input_channels, 15, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        self.model_shared=nn.Sequential(
            nn.Conv1d(15, 30, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        self.model_distinct = nn.Sequential(
            nn.Conv1d(15, 30, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        self.global_pool_shared = nn.AdaptiveAvgPool1d(1)
        self.global_pool_distinct = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(30, num_classes)

    def forward(self, x):
        x = self.model_feat(x)
        x1 = self.model_shared(x)
        x1=self.global_pool_shared(x1)
        y_shared = x1.view(x1.size(0), -1)
        x2=self.model_distinct(x)
        x2=self.global_pool_distinct(x2)
        y_distinct=x2.view(x2.size(0), -1)
        y_pred = self.fc(y_distinct)
        return y_shared, y_distinct, y_pred


class CNN_1D_V1(nn.Module):
    def __init__(self, input_channels=1, num_classes=9):
        super(CNN_1D_V1, self).__init__()
        self.model_feat = nn.Sequential(
            nn.Conv1d(input_channels, 15, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        self.model_shared = nn.Sequential(
            nn.Conv1d(15, 30, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        self.model_distinct = nn.Sequential(
            nn.Conv1d(15, 30, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        self.global_pool_shared = nn.AdaptiveAvgPool1d(1)
        self.global_pool_distinct = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(30, num_classes)

    def forward(self, x):
        x = self.model_feat(x)
        x1 = self.model_shared(x)
        x1 = self.global_pool_shared(x1)
        y_shared = x1.view(x1.size(0), -1)
        x2 = self.model_distinct(x)
        x2 = self.global_pool_distinct(x2)
        y_distinct = x2.view(x2.size(0), -1)
        y_pred = self.fc(y_distinct)
        return y_shared, y_distinct, y_pred


class PhyModel_1(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(PhyModel_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 15, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(15, 30, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(30, num_classes)

    def forward(self, x):
        inputs1 = torch.fft.fft(torch.complex(x, torch.zeros_like(x))).abs()
        positions = [10, 20, 29, 39, 49, 59, 68, 79, 89, 98, 107, 120, 147, 294, 437, 581, 732, 880, 1027, 1180, 1324,
                     137, 284, 427, 571, 722, 870, 1017, 1170, 1314]  # S1
        extracted_values = inputs1[:, :, positions]
        extracted_values = extracted_values.view(x.size(0), 30)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        y_30 = x+extracted_values
        y_pred = self.fc(y_30)
        return y_30, y_pred



