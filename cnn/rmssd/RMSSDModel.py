import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RMSSDModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)