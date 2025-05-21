import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader


class BreathClassifier(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, output_size=2):
        super(BreathClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x  # logits (do użycia z CrossEntropyLoss)


X = torch.tensor([
    [1.0, 2.0, 0.5, 70.0, 0.3],
    [0.8, 1.5, 0.4, 68.0, 0.4],
    # ...
], dtype=torch.float32)

y = torch.tensor([0, 1], dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Inicjalizacja
model = BreathClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trenowanie
for epoch in range(100):
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss = {loss.item():.4f}")

with torch.no_grad():
    sample = torch.tensor([[1.1, 2.2, 0.6, 72.0, 0.35]], dtype=torch.float32)
    logits = model(sample)
    prediction = torch.argmax(logits, dim=1).item()
    print("Predykcja:", "Wdech" if prediction == 1 else "Wydech")
