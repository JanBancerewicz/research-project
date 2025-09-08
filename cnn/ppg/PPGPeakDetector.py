import torch
import torch.nn as nn
import torch.nn.functional as F


class PPGPeakDetector(nn.Module):
    def __init__(self):
        super(PPGPeakDetector, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.conv4 = nn.Conv1d(128, 32, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.gap = nn.AdaptiveAvgPool1d(100)  # Output 100 time steps
        self.out = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1)
        x = self.gap(x)
        x = torch.sigmoid(self.out(x))
        return x.squeeze(1)  # (B, 100)
