import os.path

import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from cnn.ECG_CNN import split_into_chunks, MODEL_PATH, ECG_CNN
from cnn.train import init_model


def predict(device, model, sample):
    model.eval()
    sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(sample).squeeze(0)
        prediction = torch.sigmoid(output)
        binary_output = (prediction > 0.5).float().cpu().numpy()
    return binary_output




def get_model(device):
    if os.path.exists(MODEL_PATH):
        model = ECG_CNN().to(device)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        return model
    else:
        return init_model()

def main():
    global sig
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(device)

    df = pd.read_csv("data/R6.csv")  # Replace with your file path

    test_signals = np.array(split_into_chunks(df['ecg'].to_numpy()), dtype=np.float32)
    test_labels = np.array(split_into_chunks(df['R'].to_numpy()), dtype=np.float32)



    test_dataset = TensorDataset(torch.tensor(test_signals).unsqueeze(1), torch.tensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    s = []
    t = []
    r = []
    for i in range(len(test_signals)):
        sig = test_signals[i]
        prediction = predict(device, model, sig)
        for v in sig:
            s.append(v)

        print(f"Predykcja dla przykładowego sygnału:\n{len(prediction)}")


        for j in range(len(prediction)):
            if prediction[j] == 1:
                t.append(j + i*256)
                r.append(sig[j])

    plt.plot(s)
    plt.plot(t, r, "ro", label="Detected R-peaks")
    plt.show()



main()