import torch
import torch.nn as nn
import torch.nn.functional as F


class PPGPeakDetector(nn.Module):
    def __init__(self):
        super(PPGPeakDetector, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))
        return x.squeeze(1)  # (B, 50)
