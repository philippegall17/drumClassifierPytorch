# Drum Sample Classifier Convolutional Neural Network (CNN) using PyTorch
import torch
import numpy as np
import matplotlib.pyplot as plt

# Plotting Helpers
def _plot_grid(tensor, title):
    """Plot all feature maps in a tensor of shape (Channel, Height, Width) as a grid."""
    num_maps  = tensor.shape[0]
    cols      = 8
    rows      = int(np.ceil(num_maps / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes      = axes.flatten()
    for i in range(num_maps):
        axes[i].imshow(tensor[i].detach().cpu().numpy(), origin="lower", aspect="auto", cmap="viridis")
        axes[i].axis("off")
        axes[i].set_title(f"K{i+1}", fontsize=7)
    for j in range(num_maps, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.pause(2)
    plt.close(fig)
def _plot_mel(tensor):
    """Plot the input mel spectrogram (1, NUM_MELBANDS, time_steps)."""
    data    = tensor.squeeze().detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    img = ax.imshow(data, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("Input Mel Spectrogram", fontsize=13)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Mel bands")
    fig.colorbar(img, ax=ax, label="Normalised dB")
    plt.tight_layout()
    plt.pause(2)
    plt.close(fig)

class DrumClassifierCNN(torch.nn.Module):
    '''
    Classify drum sounds from mel spectrogram images using CNNs, weighted average kernels (filter windows) that scan the image.
    Input:          A (1 channel (from mono audio) x NUM_MELBANDS (image height) x clip_length * SAMPLERATE / HOP_LENGTH (time / image width)) "tensor" image
    Architecture:   We use K convolution kernels (small matrices of size m x m) whose entries are the weights we learn. 
                    They all scan the images top left to bottom right, learning different things due to their random initialization (Kaiming).
                    Each kernel weight-sums the m x m entries of the original image into a single weighted average value. 
                    We store all the weighted average values top left to bottom right as a new image, the so-called feature map. 
                    The feature map would be of size (x-2) * (y-2) since we cant scan outside the matrix, padding refills the borders with zeroes.
                    Before continuing, the feature map is batch-normalized, ReLUed (potential negative values replaced with 0), and MaxPooled.
                    ReLU is a simple, but nonlinear function that transforms the linear matrix multiplication setting into a nonlinear setting. 
                    Only nonlinear networks can approximate nonlinear functions (universal approximation theorem).
                    MaxPool2d(2) Scans in 2x2 blocks, keeping only the maximum value from each block, halving the spatial dimensions to make the model position-insensitive.
                    The edited feature map will be scanned again in a next "Block" step using new convolution kernels, whose weights are also trained.
                    We chose to change K per block, but not m, which could also be different in theory.
                    4 "Blocks" of pattern recognitions are repeated. We enter the original images of size x rows * y columns. Then we get:
                                    Channel                                 Kernels K   Weights per kernel  New amount of weights   Resulting Feature Map size after halving
                        Block 1:    1  (monoscale mel spectrogram image)    32          m x m = 3x3 = 9     1 * 32 * 3 * 3 = 288    x/2 rows * y/2 columns
                        Block 2:    32 (each generated feature map)         64          9                   32 * 64 * 3 * 3 = 18432 x/4  * y/4
                        Block 3:    64 (== amount of Kernels in step 2)     128         9                   64 * 128 * 9 = 73728    x/8  * y/8
                        Block 4:    128                                     128         9                   147,456                 x/16 * y/16
                    After this, the final 128 feature maps of size x/16 * y/16 that have been produced after block 4 are each resampled into 4 x 4 average-value matrices.
                    The 128 feature maps of size 4x4 are then flattened into a single vector of 128 * 4 * 4 = 2048 numbers, the model's full abstract description of the input sound.
                    These 2048 numbers are activation values, and we interpose a new fully connected layer of 256 outputs behind them for 2048 * 256 = 524,288 weights inbetween.
                    ReLU and a random dropout of 40% of the 256 activations is applied during training to avoid overfitting.
                    Finally, the last layer are the amount of different drum sample categories we have, so 256 * len(folders). Its active values are called scores or logits.
                    Softmax is applied afterwards in the inference to convert these scores into probabilities exp(s_i) / sum(exp(s_j)) summing to 1.0.
    Learning Idea:  Training and Learning is using Cross Entropy Loss, Backpropagation and the Adam Optimizer.
                    Loss:      L = -log(p_c) where p_c is the predicted probability of the correct class c (the softmaxed logit).
                    Backprop.: Chain rule applied in reverse through the computation graph: dL/dw for every weight w.
                               PyTorch records all forward operations and traverses them backwards automatically (loss.backward()).
                    Update:    Adam: w = w - learningrate * dL/dw, with learningrate (unknown hyperparameter we simply define initially) adapted per weight using gradient history.
                               Repeat for every batch until weights converge to values that minimise L.
    Other params:   Unrelated to the above logic for learning the weights themselves, there are 
                    Epochs, the amount of how often each sample has been fed to the cnn (the network has different weights each time and sees the sample differently), 
                    and the batch size, after how many samples you average the gradient for weight updating.
    '''
    kernelsize    = 3
    KernelsBlock1 = 32
    KernelsBlock2 = 64
    KernelsBlock3 = 128
    KernelsBlock4 = 128
    finallayer    = 256
    averagematriX = 4
    dropoutrate   = 0.4
    def __init__(self, num_classes, make_plots=False):
        super().__init__()     # initializes torch.nn.Module's internals, required for .parameters(), .to(cuda-device) etc. to work
        self._plotted  = False # gate: only plot for the very first forward call
        self.enableplt = make_plots

        # The individual layers, a sequential setup is much simpler (see comment below) which would work when we didn't want to plot the layers
        self.conv1   = torch.nn.Conv2d(1, self.KernelsBlock1, kernel_size=self.kernelsize, padding=1) # Conv2d scans a 3x3 filter window across the image, taking 1 input channel, producing 32 feature maps (filter outputs).
        self.bn1     = torch.nn.BatchNorm2d(self.KernelsBlock1)                                       # padding=1 pads the edges so the output stays the same height/width as the input by filling the matrix edges with zeroes.
        # Block 1 relu + pool are shared instances reused across all blocks (stateless, no weights)
        self.relu    = torch.nn.ReLU()       # nonlinearity: output = max(0, x). Without this all layers collapse into one linear operation and depth is useless.
        self.pool    = torch.nn.MaxPool2d(2) # halves height and width by keeping only the max value in each 2x2 block, giving translation invariance.
        self.conv2   = torch.nn.Conv2d(self.KernelsBlock1, self.KernelsBlock2, kernel_size=self.kernelsize, padding=1) # Next Block: 32 input channels to 64 feature maps
        self.bn2     = torch.nn.BatchNorm2d(self.KernelsBlock2)
        self.conv3   = torch.nn.Conv2d(self.KernelsBlock2, self.KernelsBlock3, kernel_size=self.kernelsize, padding=1) # Next Block: 64 to 128 feature maps
        self.bn3     = torch.nn.BatchNorm2d(self.KernelsBlock3)
        self.conv4   = torch.nn.Conv2d(self.KernelsBlock3, self.KernelsBlock4, kernel_size=self.kernelsize, padding=1) # Next Block: 128 to 128 feature maps
        self.bn4     = torch.nn.BatchNorm2d(self.KernelsBlock4)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((self.averagematriX, self.averagematriX))              # resamples each of the 128 final feature maps to a fixed 4x4 grid regardless of input width

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),                                                                             # collapses the 128 final feature maps of 4x4 dimension to one flat vector of 128*4*4 = 2048 values
            torch.nn.Linear(self.KernelsBlock4 * self.averagematriX * self.averagematriX, self.finallayer), # fully connected layer: every one of the 2048 values connects to every one of 256 outputs
            torch.nn.ReLU(),                                                                                # nonlinearity before the final output layer
            torch.nn.Dropout(self.dropoutrate),             # randomly zeroes 40% of neurons during training so the model can't rely on any single neuron to reduce overfitting
            torch.nn.Linear(self.finallayer, num_classes),  # final layer: 256 values to one score per drum class (logits, not probabilities yet)
        )

    def forward(self, x):
        # The forward pipeline. PyTorch derives the backward pass (gradient computation) automatically from this.
        if self.enableplt and not self._plotted:
            # First forward call with plots enabled: show mel + all feature maps before/after each pool.
            self._plotted = True
            _plot_mel(x[0])  # x[0]: first sample in batch, shape (1, NUM_MELBANDS, time)
            # Block 1
            x = self.relu(self.bn1(self.conv1(x)))
            _plot_grid(x[0], "Block 1: 32 feature maps before MaxPool")
            x = self.pool(x)
            _plot_grid(x[0], "Block 1: 32 feature maps after MaxPool (input / 2)")
            # Block 2
            x = self.relu(self.bn2(self.conv2(x)))
            _plot_grid(x[0], "Block 2: 64 feature maps before MaxPool")
            x = self.pool(x)
            _plot_grid(x[0], "Block 2: 64 feature maps after MaxPool (input / 4)")
            # Block 3
            x = self.relu(self.bn3(self.conv3(x)))
            _plot_grid(x[0], "Block 3: 128 feature maps before MaxPool")
            x = self.pool(x)
            _plot_grid(x[0], "Block 3: 128 feature maps after MaxPool (input / 8)")
            # Block 4
            x = self.relu(self.bn4(self.conv4(x)))
            _plot_grid(x[0], "Block 4: 128 feature maps before AvgPool")
            x = self.avgpool(x)
            _plot_grid(x[0], "Block 4: 128 feature maps after AdaptiveAvgPool (4x4)")
        else:
            # Normal path: no plots (plots disabled, or already plotted once)
            x = self.avgpool(
                    self.relu(self.bn4(self.conv4(
                    self.pool(self.relu(self.bn3(self.conv3(
                    self.pool(self.relu(self.bn2(self.conv2(
                    self.pool(self.relu(self.bn1(self.conv1(x)
                )))))))))))))))
        x = self.classifier(x) # feature vector to one score per class
        return x               # raw scores / logits, not probabilities yet
    
# Original Sequential version of DrumClassifierCNN (kept for reference) below.
# Simpler to read, but forward() cannot tap between Conv->BN->ReLU and MaxPool
# to snapshot feature maps, which is why the individual-layer version above
# was introduced. Weight key names differ (features.0, features.1 ... instead
# of conv1, bn1 ...) so saved models are NOT cross-compatible between the two.
# =============================================================================
#
# class DrumClassifierCNN(torch.nn.Module):
#     kernelsize    = 3
#     KernelsBlock1 = 32
#     KernelsBlock2 = 64
#     KernelsBlock3 = 128
#     KernelsBlock4 = 128
#     finallayer    = 256
#     averagematriX = 4
#     dropoutrate   = 0.4
#
#     def __init__(self, num_classes):
#         super().__init__() # initializes torch.nn.Module's internals, required for .parameters(), .to(cuda-device) etc. to work
#         self.features = torch.nn.Sequential(
#             torch.nn.Conv2d(1, self.KernelsBlock1, kernel_size=self.kernelsize, padding=1), # Conv2d scans a 3x3 filter window across the image, taking 1 input channel, producing 32 feature maps (filter outputs).
#             torch.nn.BatchNorm2d(self.KernelsBlock1),                                       # padding=1 pads the edges so the output stays the same height/width as the input by filling the matrix edges with zeroes.
#             torch.nn.ReLU(),                                                                # nonlinearity: output = max(0, x). Without this all layers collapse into one linear operation and depth is useless.
#             torch.nn.MaxPool2d(2),                                                          # halves height and width by keeping only the max value in each 2x2 block, giving translation invariance.
#
#             torch.nn.Conv2d(self.KernelsBlock1, self.KernelsBlock2, kernel_size=self.kernelsize, padding=1), # Next Block
#             torch.nn.BatchNorm2d(self.KernelsBlock2),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2),
#
#             torch.nn.Conv2d(self.KernelsBlock2, self.KernelsBlock3, kernel_size=self.kernelsize, padding=1),
#             torch.nn.BatchNorm2d(self.KernelsBlock3),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2),
#
#             torch.nn.Conv2d(self.KernelsBlock3, self.KernelsBlock4, kernel_size=self.kernelsize, padding=1),
#             torch.nn.BatchNorm2d(self.KernelsBlock4),
#             torch.nn.ReLU(),
#
#             torch.nn.AdaptiveAvgPool2d((self.averagematriX, self.averagematriX)), # (Technically not necessary as all spectrograms are the same width)
#         )
#
#         self.classifier = torch.nn.Sequential(
#             torch.nn.Flatten(),                                                                             # collapses the 128 final feature maps of 4x4 dimension to one flat vector of 128*4*4 = 2048 values
#             torch.nn.Linear(self.KernelsBlock4 * self.averagematriX * self.averagematriX, self.finallayer), # fully connected layer: every one of the 2048 values connects to every one of 256 outputs
#             torch.nn.ReLU(),
#             torch.nn.Dropout(self.dropoutrate),             # randomly zeroes 40% of neurons during training so the model can't rely on any single neuron to reduce overfitting
#             torch.nn.Linear(self.finallayer, num_classes),  # final layer: 256 values to one score per drum class
#         )
#
#     def forward(self, x):
#         # The forward pipeline. PyTorch derives the backward pass (gradient computation) automatically from this.
#         x = self.features(x)    # spectrogram image to abstract feature vector
#         x = self.classifier(x)  # feature vector to one score per class
#         return x                # raw scores / logits, not probabilities yet