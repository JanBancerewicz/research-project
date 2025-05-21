import os.path

import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from cnn.ecg.ECG_CNN import split_into_chunks, MODEL_PATH, ECG_CNN
from cnn.ecg.train import init_model


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
        print("MODEL LOADED")
        return model
    else:
        return init_model()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(device)

    df = pd.read_csv( "data/r/R19.csv")  # Replace with your file path

    test_signals = np.array(split_into_chunks(df['ecg'].to_numpy()), dtype=np.float32)
    test_labels = np.array(split_into_chunks(df['R'].to_numpy()), dtype=np.float32)



    test_dataset = TensorDataset(torch.tensor(test_signals).unsqueeze(1), torch.tensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    correct = 0
    all_pred = 0
    all_peaks = 0
    missed = 0
    for i in range(len(test_signals)):
        sig = test_signals[i]
        prediction = predict(device, model, sig)
        for j in range(len(prediction)):
            if prediction[j] == test_labels[i][j] and test_labels[i][j] == 1:
                correct += 1
            if test_labels[i][j] == 1:
                all_peaks += 1
            if prediction[j] == 1:
                all_pred += 1
            if prediction[j] == 0 and test_labels[i][j] == 1:
                missed += 1

    p = correct / all_peaks * 100
    p2 = max( (all_pred / all_peaks * 100) - 100, 0)
    p3 = missed / all_peaks * 100

    print(f"Accuracy: {p:.2f}%")
    print(f"Additional: {p2:.2f}%")
    print(f"Missed: {p3:.2f}%")
    plot_ecg(device, model, test_signals, test_labels)




def plot_ecg(device, model, test_signals, test_labels):
    s = []
    t = []
    r = []

    t_t = []
    r_t = []
    for i in range(20):
        sig = test_signals[i]
        prediction = predict(device, model, sig)
        for v in sig:
            s.append(v)



        for j in range(len(prediction)):
            if prediction[j] == 1:
                t.append(j + i*256)
                r.append(sig[j])
            if test_labels[i][j] == 1:
                t_t.append(j + i*256)
                r_t.append(sig[j])

    plt.plot(s)
    plt.plot(t, r, "ro", label="Detected R-peaks")

    plt.figure()
    plt.plot(s)
    plt.plot(t_t, r_t, "ro", label="Detected R-peaks")
    plt.show()