# Analysis of Heart Rate Variability Using Mobile Devices and Machine Learning

This repository contains the research code for the project **“Analysis of heart rate variability using mobile devices and machine learning”**, which investigates heart rate variability (HRV) based on electrocardiographic (ECG) and photoplethysmographic (PPG) signals acquired using widely available mobile devices.

The project combines classical signal processing with deep learning models for peak detection and compares the resulting HRV indicators between ECG and PPG.

> This repository contains the Python side of the project (signal processing, modeling, analysis).  
> The companion Android app for PPG acquisition is available at:  
> **https://github.com/JanBancerewicz/PPGbetter**

## Research context and contributors

This repository originates from a research project carried out at Gdańsk University of Technology (Politechnika Gdańska).  
The work resulted in the following peer-reviewed publication in TASK Quarterly:

> Jan Kosma Bancerewicz, Julian Jerzy Kotłowski, Ostap Lozovyy,  
> Julia Beata Morawska, Mateusz Rzęsa,  
> **“Analysis of heart rate variability using mobile devices and machine learning”**,  
> TASK Quarterly, Vol. 30, No. 4, 2025.  # TODO waiting for a review
> PDF: [Analysis_of_heart_rate_variability_using_mobile_devices_and_machine_learning.pdf](https://github.com/JanBancerewicz/research-project/blob/doc/Analysis_of_heart_rate_variability_using_mobile_devices_and_machine_learning%20(compiled).pdf)


The project and this codebase were developed collaboratively by:

- **Jan Bancerewicz** – (https://github.com/JanBancerewicz)
- **Julian Jerzy Kotłowski** – (https://github.com/julkot1)
- **Ostap Lozovyy** – (https://github.com/agoneiro)
- **Julia Beata Morawska** – (https://github.com/jmorawska03)   
- **Mateusz Rzęsa** – (https://github.com/L1itrer)

If you use this repository or build upon this work, please reference the above publication (see also the [Citation](#citation) section).

---


## Table of Contents

1. [Research context and contributors](#research-context-and-contributors)  
2. [Project overview](#project-overview)  
3. [Repository structure](#repository-structure)
4. [Reproducing the experiments](#reproducing-the-experiments)  
5. [Entry points](#entry-points)   
6. [Signal processing pipeline](#signal-processing-pipeline)  
7. [Neural models](#neural-models)  
8. [HRV indicators and PTT](#hrv-indicators-and-ptt)   
9. [Companion Android application](#companion-android-application)  
10. [Citation](#citation)
11. [License](#license)

---

## Project overview

The goal of this project is to:

- Acquire ECG and PPG signals using **mobile-grade hardware** (Polar H10 chest strap and an Android smartphone).  
- Apply **filtering** and **peak detection** to obtain beat-to-beat intervals.  
- Compare **classical algorithms** (e.g. Pan–Tompkins for ECG) with **deep neural networks** for peak detection.  
- Compute and compare **HRV indicators** obtained from ECG (RR intervals) and PPG (IBI – inter-beat intervals).  
- Assess the feasibility of using mobile PPG as a reliable alternative to ECG for HRV analysis in controlled conditions.

The models achieve high peak-detection performance (F1 ≈ 0.98 for ECG R-peaks and ≈ 0.98 for PPG peaks) and show that, under proper recording conditions, PPG can approximate ECG-based HRV metrics while being more sensitive to motion artifacts.

---

## Repository structure

Top-level layout of the `main` branch:

- `appgui/`  
  Desktop / GUI utilities for visualizing signals, peaks and HRV indicators. Useful for manual inspection and demonstration of the processing pipeline.

- `cnn/`  
  Deep learning models and training scripts for:
  - ECG R-peak detection (convolutional + recurrent network),
  - PPG peak detection (convolutional residual network with SE module),
  as described in the paper.

- `comparison/`  
  Scripts for comparing:
  - classical peak detection (e.g. Pan–Tompkins, local maxima rules),
  - neural models,
  - derived HRV metrics from ECG vs PPG.

- `data/`  
  Helper data, configuration, and/or example CSV files.  
  The full recordings used in the paper are not necessarily included here due to size and privacy constraints.

- `old/`  
  Legacy or experimental code kept for reference.

- Top-level Python modules:
  - `data_processing.py` – main signal processing and analysis routines
  - `filtr.py` – filter design and helper functions
  - `pan_tompkins.py` – classical Pan–Tompkins implementation
  - `ppg.py` – PPG-specific processing
  - `r_neural.py` – ECG R-peak neural model code
  - `ngui.py` – graphical front-end / main interactive entry point (see below)
  - `tests.py`, `testxd.py` – basic tests / experimental scripts
  - `requirements.txt` – Python dependencies
  - `data.zip` – example data archive

---

## Reproducing the experiments

> The exact commands may depend on how you organize your environment and data paths.  
> The steps below describe the intended workflow; adapt them to your local setup.

### 1. Set up the environment

Create and activate a virtual environment, then install the dependencies, for example:

- `python -m venv venv`  
- (Linux/macOS) `source venv/bin/activate`  
- (Windows PowerShell) `.\venv\Scripts\Activate.ps1`  
- `pip install --upgrade pip`  
- `pip install -r requirements.txt`  

### 2. Install CUDA toolkit (for training and GPU inference)

Training the neural models and running GPU-accelerated inference assumes:

- a CUDA-capable GPU,
- a matching CUDA toolkit / drivers installed on the system,
- a PyTorch build compiled with CUDA support.

If:

- training scripts fail with errors such as `CUDA error`, `CUDA driver not found`, or `torch.cuda.is_available() == False`,  
  then the issue is most likely missing or misconfigured CUDA/GPU support.
- the code runs but the GUI or evaluation scripts complain about missing `.pth` model files,  
  you first need to train the models (see step 4) or provide pre-trained weights.

On systems without a GPU, you can still run the classical pipeline and, with appropriate code changes, train or evaluate models on CPU (significantly slower).

To check whether CUDA is available run `cuda_check.py` script.

### 3. Prepare data

Extract the example dataset archive and place its contents in the `data/` directory:

- ensure that `data/data.zip` exists in your clone,
- extract it so that raw/processed CSV files are available under `data/`


### 4. Train the neural models

Train both ECG and PPG peak-detection models, both can be trained using `ngui.py` entry point.
- ensure that `ppg_data.csv` and `data/ecg/ECG2.csv` as well as other ECG files exists.
- `ppg_data.csv` is crucial in training PPG model
- `data/ecg/*.csv` is crucial in training ECG model


### 5. Evaluate and compare

Run inference on held-out recordings and compare:
- classical vs neural peak detection (precision/recall/F1)
- ECG- vs PPG-based HRV metrics (Mean, SDNN, RMSSD)
- PTT statistics (mean, std)
- to access those metrics, run `ngui.py`, `pan_tompkins.py` and Python files from `comparison/` directory.

---

## Entry points

The repository exposes several executable scripts / entry points. The most important is `ngui.py`.

- **`ngui.py`** — Main GUI / visualizer entry point. Launches an interactive interface to:
  - load recordings (ECG/PPG CSV),
  - establish a WebSocket connection with the **PPGbetter** (https://github.com/JanBancerewicz/PPGbetter) Android app to stream live PPG signals, which are then used as input for real-time heart-rate and HRV analysis,
  - experiment with filtering and peak detection (classical vs. neural),
  - visually inspect signals and export results (peaks, IBI/RR, HRV).
  - Quick start:
    ```bash
    python ngui.py
    ```

- **`pan_tompkins.py`** — Classical Pan–Tompkins implementation for ECG R-peak detection.

- **`ecg_ai_vs_pan_tompkins.py`** – evaluates the ECG neural peak-detection model against the Pan–Tompkins algorithm (used here as the ECG reference), reporting detection metrics and visualizing differences between both methods.

- **`ppg_ai_vs_real.py`** – evaluates the PPG neural peak-detection model against reference peaks obtained with NeuroKit (used here as the PPG reference), computing detection metrics and visualizing agreement/discrepancies between model predictions and ground truth.

- **`comparePPGECG.py`** – compares paired ECG and PPG recordings, aligning them in time and generating summary statistics/plots showing how the two signals and their derived metrics relate to each other.

- **`cuda_check.py`** - Script to check whether CUDA is available.
---

## Signal processing pipeline

The processing chain is similar for ECG and PPG but uses different band-pass ranges:

1. **Filtering**
   - ECG:
     - 5th-order digital **Butterworth band-pass** filter,
     - Passband: **0.5–45 Hz**,
     - Implemented with NeuroKit2 / SciPy-like functions,
     - Suppresses baseline wander and high-frequency noise.
   - PPG:
     - 4th-order digital **Butterworth band-pass** filter,
     - Passband: **0.5–5 Hz**,
     - Implemented in SciPy,
     - Reduces movement-related low-frequency components and device/optical noise.

2. **Classical peak detection**
   - ECG:
     - **Pan–Tompkins algorithm**:
       - band-pass filtering, differentiation, squaring, moving window integration,
       - local maxima thresholding to detect R-peaks.
   - PPG:
     - Local comparison of three samples (center > neighbors) and refractory interval rule (≥ 600 ms between peaks) to suppress spurious detections.

3. **Neural peak detection**
   - ECG:
     - CNN + LSTM network operating on 1D windows (256 samples):
       - multiple 1D convolutional layers with BatchNorm + LeakyReLU,
       - max-pooling along time,
       - unidirectional LSTM,
       - fully connected head producing a probability for each sample being an R-peak.
   - PPG:
     - 1D convolutional residual network on 50-sample segments:
       - initial Conv1D + GELU + BatchNorm,
       - four residual blocks with varying kernel sizes,
       - SE (squeeze-and-excitation) module to reweight channels,
       - final Conv1D + fully-connected layers + sigmoid output for local peak probability.

4. **Postprocessing**
   - Transform model outputs (probability vectors) into discrete peak indices.
   - Enforce physiological constraints (minimum distance between peaks).
   - Pair neural detections with reference peaks (for evaluation).

---

## Neural models

### ECG R-peak detection model

- Input: 1D ECG segments of length 256 samples.  
- Architecture (high-level):
  - 4 × Conv1D blocks:
    - channels from 1 → 16 → 32 → 64 → 128,
    - kernels: size 5 / 5 / 3 / 3 with appropriate padding,
    - each followed by BatchNorm1d + LeakyReLU.
  - MaxPooling1D to compress the time dimension.
  - Unidirectional LSTM on the resulting [channels × time] representation.
  - Fully connected layers to produce a 256-element output (per-sample logits).
  - Sigmoid / thresholding for binary classification.

- Training:
  - Labels generated from Pan–Tompkins detections on filtered ECG.
  - Loss: binary cross-entropy with logits.
  - Optimizer: Adam, learning rate 1e-4.
  - Mini-batch training on Polar H10 recordings.

- Evaluation:
  - Accuracy ≈ **97%**.
  - F1 score for R-peak detection ≈ **0.98**.
  - Very low rate of false R-peaks and missed beats on the test subset.

### PPG peak detection model

- Input: 1D PPG segments of length 50 samples (normalized to [-1, 1]).  
- Architecture (high-level):
  - Conv1D (1 → 32 channels, kernel size 7, GELU + BatchNorm).
  - Residual blocks:
    - 32 → 64 (kernel 9),
    - 64 → 128 (kernel 5),
    - 128 → 128 (kernel 3),
    - 128 → 128 (kernel 7),
    - each with BatchNorm, GELU, dropout.
  - SE block on 128 channels.
  - Conv1D 128 → 64 (kernel 1).
  - Fully connected layers 128 → 64 → 32 with GELU.
  - Sigmoid output (per-sample peak probability).

- Evaluation (sample-wise classification):
  - Accuracy: **98.17%**
  - F1 score: **0.9774**
  - Confusion matrix: TN = 9609, TP = 375, FP = 9, FN = 7

- Evaluation (event-wise peak detection on continuous PPG waveforms):
  - True Positives = 61, False Positives = 15, False Negatives = 11
  - Precision ≈ 0.80, Sensitivity ≈ 0.85, F1 ≈ 0.82

These results indicate a high level of agreement between the network and the reference PPG peaks. Most false detections are related to signal artifacts and the limitations of short-segment analysis rather than systematic model errors.

---

## HRV indicators and PTT

Once peaks are detected, the code computes:

### HRV (per signal independently)

From ECG:
- RR intervals (time between successive R-peaks).

From PPG:
- IBI – Inter-Beat Intervals (time between successive PPG peaks).

From these intervals:

- **Mean RR / IBI** – average interval duration.  
- **SDNN** – standard deviation of all NN intervals (overall HRV).  
- **RMSSD** – root mean square of successive differences (short-term HRV).

A dynamic window (e.g. 60 s) is used to compute these metrics over time to mimic online monitoring.

### Pulse Transit Time (PTT)

Using synchronized ECG and PPG:

- Convert ECG relative times to UNIX timestamps.
- Align ECG R-peaks and PPG peaks on the same absolute time axis.
- For each R-peak, assign the closest PPG peak in chronological order.
- Compute:
  \[
  \text{PTT} = t_{\text{PPG}} - t_{\text{ECG}}
  \]
- Extract descriptive statistics (mean, standard deviation) to characterize the delay between electrical activation and optical pulse arrival.


---
## Companion Android application

The PPG acquisition app used in this project is open-source:

**PPGbetter** – Android application for camera-based PPG recording and real-time visualization:  
https://github.com/JanBancerewicz/PPGbetter

The app:

- uses the rear camera and LED flash to acquire a fingertip PPG signal,
- displays the raw PPG waveform and instantaneous heart rate in real time,
- can record measurement sessions locally and export them as CSV files,
- records timestamps for respiration-related events (e.g. guided breathing, user markers),
- streams averaged luma values over a WebSocket connection to a desktop/server component,  
  where they are stored as CSV and processed by the code in this repository (peak detection, HRV, PTT).

PPGbetter can be used in **live streaming mode** – connect PPGbetter to the machine running this code (same network, configured IP/port), stream the PPG signal via WebSocket, and use tools such as `ngui.py` to visualize and analyze the signal and derived HR/HRV metrics in real time.

For installation instructions, permissions and configuration details (e.g. server address, WebSocket port), refer to the README in the PPGbetter repository.

---

## Citation

If you use this code or ideas in your research, please cite the original paper:

J. K. Bancerewicz, J. J. Kotłowski, O. Lozovyy, J. B. Morawska, M. Rzęsa,  
**“Analysis of heart rate variability using mobile devices and machine learning”**,  
TASK Quarterly, Vol. 30, No. 4, 2025.  # TODO waiting for a review

---

## License

This repository is provided for research and educational purposes.  
Before using the code in commercial products or medical applications, review the license terms contained in this GitHub project.
