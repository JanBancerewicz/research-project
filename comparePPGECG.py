import pandas as pd
import matplotlib.pyplot as plt

def normalize(signal):
    min_val = signal.min()
    max_val = signal.max()
    return 2 * (signal - min_val) / (max_val - min_val) - 1

dfEcg = pd.read_csv('ecg_data_aligned.csv')
dfPgg = pd.read_csv('ppg_data_aligned.csv')

# Normalize signals to [-1, 1]
ecg_norm = normalize(dfEcg['ecg'])
ppg_norm = normalize(dfPgg['ppg'])

plt.figure(figsize=(10, 4))
plt.plot(dfEcg['time'], ecg_norm, label='ECG (normalized)')
plt.plot(dfPgg['time'], ppg_norm, label='PPG (normalized)')
plt.xlabel('Time (unix)')
plt.ylabel('Normalized ECG & PPG Signal')
plt.title('ECG Signal vs PPG Signal (Normalized)')
plt.legend()
plt.tight_layout()
plt.show()
