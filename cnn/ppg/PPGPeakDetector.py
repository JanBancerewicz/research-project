import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.gelu(self.bn2(self.conv2(out)))
        out += identity
        return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (B, C, L)
        y = x.mean(-1)  # (B, C)
        y = F.gelu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.unsqueeze(-1)  # (B, C, 1)
        return x * y


class PPGPeakDetector(nn.Module):
    def __init__(self):
        super(PPGPeakDetector, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.res1 = ResidualBlock(32, 64, kernel_size=9, dropout=0.25)
        self.res2 = ResidualBlock(64, 128, kernel_size=5, dropout=0.25)
        self.res3 = ResidualBlock(128, 128, kernel_size=3, dropout=0.25)
        self.res4 = ResidualBlock(128, 128, kernel_size=7, dropout=0.25)
        self.se = SEBlock(128)
        self.dropout = nn.Dropout(0.4)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.gap = nn.AdaptiveAvgPool1d(100)
        self.gmp = nn.AdaptiveMaxPool1d(100)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B, 1, L)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.se(x)
        x = self.dropout(x)
        x = F.gelu(self.bn2(self.conv2(x)))
        avg_pool = self.gap(x)  # (B, 64, 100)
        max_pool = self.gmp(x)  # (B, 64, 100)
        x = torch.cat([avg_pool, max_pool], dim=1)  # (B, 128, 100)
        x = x.permute(0, 2, 1)  # (B, 100, 128)
        x = F.gelu(self.fc1(x))  # (B, 100, 64)
        x = F.gelu(self.fc2(x))  # (B, 100, 32)
        x = self.out(x)  # (B, 100, 1)
        x = torch.sigmoid(x)
        return x.squeeze(-1)  # (B, 100)
