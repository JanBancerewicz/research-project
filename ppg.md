# NEW PPG DETECTION CNN
Accuracy: 98.17%
Additional: 0.52%
Missed: 1.83%

Confusion matrix:
[[9609    9]
 [   7  375]]

F1-score: 0.9774

```[run_ai_ppg] AI model detected 53 peaks in total.
Number of peaks (AI): 71
Number of peaks (Validated/Reference): 72

--- Accuracy: AI vs Real ---
True Positives: 64
False Positives: 7
False Negatives: 8
Precision: 0.901
Recall:    0.889
F1-score:  0.895
(tolerance = 8 samples = 264 ms)
```
## Network Architecture

This network is a deep 1D convolutional neural network designed for robust PPG peak detection. It consists of:

- An initial convolutional layer with batch normalization and GELU activation.
- Four stacked residual blocks with increasing channel depth and varying kernel sizes, each followed by dropout for regularization.
- A squeeze-and-excitation (SE) block for channel-wise attention, helping the network focus on informative features.
- A dropout layer for further regularization.
- A 1x1 convolution and batch normalization to reduce channel dimensionality.
- Both adaptive average pooling and max pooling to capture diverse temporal features, concatenated for richer representations.
- Two fully connected layers with GELU activation for feature refinement.
- A final output layer with sigmoid activation, producing a probability for each time step in the segment.

This architecture leverages deep residual learning, attention mechanisms, and multi-scale feature extraction to achieve high accuracy in PPG peak detection.

### Architecture Table

| Layer                | Type                | Channels/Units | Kernel Size | Activation | Other                |
|----------------------|---------------------|---------------|-------------|------------|----------------------|
| Input                | -                   | 1             | -           | -          | -                    |
| Conv1                | Conv1d              | 32            | 7           | GELU       | BatchNorm            |
| Residual Block 1     | ResidualBlock       | 64            | 9           | GELU       | Dropout 0.25         |
| Residual Block 2     | ResidualBlock       | 128           | 5           | GELU       | Dropout 0.25         |
| Residual Block 3     | ResidualBlock       | 128           | 3           | GELU       | Dropout 0.25         |
| Residual Block 4     | ResidualBlock       | 128           | 7           | GELU       | Dropout 0.25         |
| SE Block             | Squeeze-Excitation  | 128            | -           | GELU/Sigmoid| Channel Attention    |
| Dropout              | Dropout             | -             | -           | -          | p=0.4                |
| Conv2                | Conv1d              | 64            | 1           | GELU       | BatchNorm            |
| Adaptive Pooling     | Avg+Max Pool1d      | 64+64         | -           | -          | Output size 100      |
| FC1                  | Linear              | 64            | -           | GELU       | -                    |
| FC2                  | Linear              | 32            | -           | GELU       | -                    |
| Output               | Linear              | 1             | -           | Sigmoid    | -                    |

### ResidualBlock

A ResidualBlock is a building block for deep neural networks that helps mitigate the vanishing gradient problem by allowing the input to bypass (shortcut) the main convolutional layers. It consists of two convolutional layers with batch normalization and GELU activation, followed by dropout for regularization. If the input and output channels differ, a 1x1 convolution is used for the shortcut connection. The output is the sum of the shortcut and the processed input.

| Layer         | Type         | Channels      | Kernel Size | Activation | Other         |
|---------------|--------------|--------------|-------------|------------|---------------|
| Conv1    SEBlock     | Conv1d       | out_channels | kernel_size | GELU       | BatchNorm     |
| Dropout       | Dropout      | -            | -           | -          | p=dropout     |
| Conv2         | Conv1d       | out_channels | kernel_size | GELU       | BatchNorm     |
| Shortcut      | Conv1d/Identity| out_channels | 1/-         | -          | If needed     |
| Add           | -            | -            | -           | -          | Residual sum  |

**Term descriptions:**
- **Shortcut:** A direct path that allows the input to bypass one or more layers, typically implemented as an identity mapping or a 1x1 convolution to match dimensions. Used in residual blocks to help gradients flow and stabilize training.
- **Residual sum:** The operation of adding the shortcut (input) to the output of the block, enabling residual learning.

### SEBlock

A Squeeze-and-Excitation (SE) Block is a channel-wise attention mechanism. It works by globally averaging each channel, passing the result through two fully connected layers with GELU and sigmoid activations, and then scaling the original input channels by these learned weights. This helps the network focus on the most informative features.

| Step              | Type      | Output Shape         | Activation | Other                |
|-------------------|-----------|---------------------|------------|----------------------|
| Global Avg Pool   | Mean      | (B, C)              | -          | Across length        |
| FC1               | Linear    | (B, C//reduction)   | GELU       |                      |
| FC2               | Linear    | (B, C)              | Sigmoid    |                      |
| Rescale           | Multiply  | (B, C, L)           | -          | Channel-wise scaling |

**Term descriptions:**
- **FC1:** The first fully connected (linear) layer in a block, often used to reduce or transform feature dimensions.
- **Rescale:** Multiplying the input tensor by learned weights (from the SE block) to emphasize important channels.
- **Global Avg Pool:** A pooling operation that computes the mean value across the temporal dimension for each channel, summarizing the signal.
- **Sigmoid:** An activation function that squashes values to the range [0, 1], often used for probabilities or gating mechanisms.
