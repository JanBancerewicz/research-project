# NEW PPG DETECTION CNN
- Accuracy: 98.95%
- Additional: 0.26%
- Missed: 1.05%

Confusion matrix:
[[9613    5]
 [   4  378]]

```[run_ai_ppg] AI model detected 53 peaks in total.
Number of peaks (AI): 27
Number of peaks (Pan-Tompkins): 72
Number of peaks (Validated/Reference): 72

--- Accuracy: AI vs Pan-Tompkins ---
True Positives: 25
False Positives: 2
False Negatives: 47
Precision: 0.926
Recall:    0.347
F1-score:  0.505
(tolerance = 5 samples = 165 ms)

```

| Warstwa              | Typ              | Parametry                         | Rozmiar wejścia → wyjścia (kanały) | Aktywacja       |
|----------------------|------------------|-----------------------------------|-------------------------------------|-----------------|
| `conv1` + `bn1`      | Conv1d + BN      | in=1, out=32, kernel=7, pad=3     | (B, 1, L) → (B, 32, L)             | LeakyReLU(0.1)  |
| `conv2` + `bn2`      | Conv1d + BN      | in=32, out=64, kernel=5, pad=2    | (B, 32, L) → (B, 64, L)            | LeakyReLU(0.1)  |
| `conv3` + `bn3`      | Conv1d + BN      | in=64, out=128, kernel=3, pad=1   | (B, 64, L) → (B, 128, L)           | LeakyReLU(0.1)  |
| `dropout`            | Dropout          | p=0.3                             | (B, 128, L) → (B, 128, L)          | —               |
| `conv4` + `bn4`      | Conv1d + BN      | in=128, out=32, kernel=1          | (B, 128, L) → (B, 32, L)           | LeakyReLU(0.1)  |
| `gap`                | AdaptiveAvgPool1d| output_size=100                   | (B, 32, L) → (B, 32, 100)          | —               |
| `out`                | Conv1d           | in=32, out=1, kernel=1            | (B, 32, 100) → (B, 1, 100)         | Sigmoid         |
| `squeeze(1)`         | Tensor op        | usuwa wymiar kanałów              | (B, 1, 100) → (B, 100)             | —               |
