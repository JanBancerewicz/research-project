# python
import os
import time
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cnn.ppg.PPGPeakDetector import PPGPeakDetector
from cnn.ppg.data import get_or_train_model, PPGDirectoryDataset

# --- konfiguracja (dostosuj) ---
ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "ppg_peak_model.pth")
DATA_DIR = os.path.join(ROOT, "cnn", "ppg", "train_data")
SEGMENT_LENGTH = 100
EPOCHS = 2
BATCH_SIZE = 64
LR = 1e-3
NUM_WORKERS = 0  # Windows: bezpieczne ustawienie; możesz podnieść do 2-4 jeśli działa

# --- szybkie sprawdzenie CUDA ---
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("device count:", torch.cuda.device_count())
        print("current device:", torch.cuda.current_device())
        print("device name:", torch.cuda.get_device_name(0))
        print("memory allocated (B):", torch.cuda.memory_allocated(0))
        print("memory reserved  (B):", torch.cuda.memory_reserved(0))
    except Exception as e:
        print("CUDA info error:", e)

# opcjonalnie: pokaż nvidia-smi jeśli dostępne
try:
    out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, universal_newlines=True)
    print("nvidia-smi (first lines):\n", "\n".join(out.splitlines()[:6]))
except Exception:
    pass

# ustawienia wydajności
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==== pobierz / wytrenuj model (get_or_train_model zrobi zapis jeśli trzeba) ====
print(f"Model path: {MODEL_PATH}")
model = get_or_train_model(
    model_path=MODEL_PATH,
    data_dir=DATA_DIR,
    segment_length=SEGMENT_LENGTH,
    max_segments=10000,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    max_files=None,
)

# Przenieś model na device
model.to(device)
print("Model device after .to(device):", next(model.parameters()).device)

# ==== przygotuj dataset / dataloader ====
# Spróbuj użyć PPGDirectoryDataset jeśli dostępny i zawiera pliki
dataset = None
try:
    if os.path.isdir(DATA_DIR) and any(f.endswith('.csv') for f in os.listdir(DATA_DIR)):
        dataset = PPGDirectoryDataset(DATA_DIR, segment_length=SEGMENT_LENGTH)
        print(f"Loaded dataset from {DATA_DIR}, samples={len(dataset)}")
    else:
        print(f"No csv files in {DATA_DIR} - fallback to synthetic data for quick test")
except Exception as e:
    print("Dataset load error:", e)

# Fallback: syntetyczny dataset (tylko do testów)
if dataset is None or len(dataset) == 0:
    # ustal input shape (N, C, L)
    in_channels = 1
    # spróbuj wywnioskować z pierwszej konwolucji modelu
    for name, p in model.named_parameters():
        if p.ndim >= 3:
            in_channels = p.shape[1]
            break
    N = 1024
    X = torch.randn(N, in_channels, SEGMENT_LENGTH)
    # output shape: model zwraca (B, L_out) czyli (B, SEGMENT_LENGTH) w tej architekturze
    with torch.no_grad():
        try:
            sample_out = model(X[:1].to(device)).cpu()
            out_shape = tuple(sample_out.shape[1:])
        except Exception:
            out_shape = (SEGMENT_LENGTH,)
    y = torch.randn(N, *out_shape)
    dataset = TensorDataset(X, y)
    print("Using synthetic dataset", X.shape, y.shape)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))

# ==== trening (krótkie epoki testowe) ====
scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# model.forward kończy sigmoidem -> używamy BCELoss
loss_fn = torch.nn.BCELoss()

print(f"Starting quick training loop on device={device} for {EPOCHS} epoch(s)")
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    t0 = time.time()
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            preds = model(X_batch)
            # upewnij się że y_batch ma odpowiedni kształt
            if preds.shape != y_batch.shape:
                # broadcast/reshape jeśli trzeba
                try:
                    y_batch = y_batch.view(preds.shape)
                except Exception:
                    y_batch = y_batch.expand_as(preds)
            loss = loss_fn(preds, y_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item() * X_batch.size(0)
    dur = time.time() - t0
    avg_loss = epoch_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.6f} - time: {dur:.2f}s")

# ==== szybki benchmark forward ====
model.eval()
with torch.no_grad():
    try:
        sample_x = torch.randn(32, in_channels, SEGMENT_LENGTH).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            _ = model(sample_x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        print("avg forward ms:", (time.time() - t0) / 50 * 1000)
    except Exception as e:
        print("Benchmark forward error:", e)

print("Quickscript finished.")
