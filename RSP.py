import neurokit2
import pandas as pd
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# === 1. Wczytaj dane ===

import neurokit2 as nk
import matplotlib.pyplot as plt


data = pd.read_csv("data/R1.csv")
ecg = data['ecg'].values

fs = 130  # Hz

rsp = nk.ecg_rsp(ecg, sampling_rate=fs, method='vangent2019')
rsp2 = nk.ecg_rsp(ecg, sampling_rate=fs, method='soni2019')
rsp3 = nk.ecg_rsp(ecg, sampling_rate=fs, method='charlton2016')

plt.figure(figsize=(12, 4))
plt.plot(rsp, label="Oszacowany sygnał oddechu (vangent2019)")
plt.plot(rsp2, label="Oszacowany sygnał oddechu (soni2019)")
plt.plot(rsp3, label="Oszacowany sygnał oddechu (charlton2016)")
plt.xlabel("Próbki")
plt.ylabel("Amplituda (jednostki względne)")
plt.title("Sygnał oddechu wyodrębniony z EKG (ecg_rsp)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
