from torch import nn
import torch.nn.functional as F


class EDR_CNN(nn.Module):
    def __init__(self, input_length=100):
        super(EDR_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.pool3 = nn.AdaptiveAvgPool1d(1)  # sprowadzenie do [batch, 128, 1]

        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)  # regresja BPM

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))   # [B, 32, ~49]
        x = self.pool2(F.relu(self.conv2(x)))   # [B, 64, ~24]
        x = self.pool3(F.relu(self.conv3(x)))   # [B, 128, 1]
        x = x.view(x.size(0), -1)               # [B, 128]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
