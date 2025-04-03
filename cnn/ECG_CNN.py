import numpy as np
from torch import nn
import torch.nn.functional as F

MODEL_PATH = "cnn/ecgcnn.pth"


class ECG_CNN(nn.Module):
    def __init__(self, window_size=256, dropout_prob=0.5):
        super(ECG_CNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_prob)

        flattened_size = (window_size // 8) * 64
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, window_size)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x  # Output shape: (batch_size, window_size)

# class ECG_CNN(nn.Module):
#     def __init__(self, window_size=256):
#         super(ECG_CNN, self).__init__()
#         self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
#         self.bn1 = nn.BatchNorm1d(16)
#         self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
#         self.bn2 = nn.BatchNorm1d(32)
#         self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm1d(64)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#
#         self.fc1 = nn.Linear((window_size // 8) * 64, 128)
#         self.fc2 = nn.Linear(128, window_size)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

def split_into_chunks(arr, chunk_size=256):
    arr = np.array(arr, dtype=float)  # Konwersja na numpy array
    num_full_chunks = len(arr) // chunk_size  # Liczba pełnych chunków
    return np.split(arr[:num_full_chunks * chunk_size], num_full_chunks)  # Dzielimy tylko pełne chunk

