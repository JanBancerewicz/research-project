import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import sys
import os

# Importy z Twojego projektu (dostosowane do struktury folderów)
try:
    from cnn.ppg.data import get_or_train_model, PPGDirectoryDataset
except ImportError:
    # Fallback path fix
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from cnn.ppg.data import get_or_train_model, PPGDirectoryDataset


def check_cuda_training():
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available. Training will be slow (CPU).")
        return

    device = torch.device("cuda")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    # 1. Ładowanie modelu (korzysta z Twojego naprawionego data.py)
    # Używamy dummy path, bo chcemy tylko załadować architekturę lub wytrenować na chwilę
    model_path = "cuda_test_model.pth"

    # Uwaga: Tutaj data_dir jest ignorowany przez Twój naprawiony skrypt, więc wpisuje cokolwiek
    print("\n--- Initializing Data & Model ---")
    try:
        # Wywołujemy get_or_train_model ale z parametrami, które tylko przygotują dane
        # Wczytujemy model, ale nie trenujemy go wewnątrz funkcji (chyba że nie istnieje)
        # Hack: ustawiamy epochs=0 jeśli funkcja na to pozwala, lub po prostu pobieramy Dataset ręcznie

        # Podejście ręczne (bezpieczniejsze dla testu):
        from cnn.ppg.data import load_ppg_segments_from_directory, PPGDirectoryDataset
        from cnn.ppg.PPGPeakDetector import PPGPeakDetector

        # Ustalamy ścieżkę do danych tak jak w data.py
        current_dir = os.path.dirname(os.path.abspath(__file__))  # research-project
        data_dir = os.path.join(current_dir, "cnn", "ppg", "train_data")

        print(f"Loading data from: {data_dir}")

        # ZMIANA TUTAJ: segment_length=100 (było 50)
        dataset = PPGDirectoryDataset(data_dir, segment_length=100, max_files=2)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        print(f"Dataset loaded. Samples: {len(dataset)}")

        model = PPGPeakDetector().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()  # To jest Twoja funkcja straty

        print("\n--- Starting CUDA Training Loop Test ---")
        model.train()

        start = time.time()
        for i, (x_batch, y_batch) in enumerate(dataloader):
            # Przeniesienie na GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # --- ZWYKŁY TRENING (BEZ AUTOCAST) ---
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Batch {i}: Loss = {loss.item():.4f} (CUDA is working!)")

            if i >= 50:  # Przerwij po 50 batchach, to tylko test
                break

        end = time.time()
        print(f"\n✅ SUCCESS! CUDA training loop finished.")
        print(f"Processed batches in {end - start:.2f} seconds.")

    except Exception as e:
        print(f"\n❌ ERROR during CUDA check: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_cuda_training()