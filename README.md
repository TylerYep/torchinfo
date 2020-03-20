## torch-summary

This is a rewritten version of the original torchsummary and torchsummaryX projects by @sksq96 and @nmhkahn.
There are quite a few pull requests on the original project (which hasn't been updated in over a year), so I decided to take a stab at improving and consolidating some of its features.

This version now has support for:
- RNNs, LSTMs, and other recursive layers
- Tree-Branch output to explore model layers using specific depths
- Verbose mode to show specific weights and bias layers
- More comprehensive testing using pytest

Keras has a neat API to view the visualization of the model which is very helpful while debugging your network. In this project, we attempt to do the same in PyTorch. The goal is to provide information complementary to what is provided by `print(your_model)` in PyTorch.

### Usage

- `pip install torch-summary`
or
- `git clone https://github.com/tyleryep/torch-summary.git`

Notice the dash in torch-summary!

```python
from torchsummary import summary
summary(your_model, input_size=(C, H, W))
```

- Note that the `input_size` is required to make a forward pass through the network.

### Examples

#### CNN for MNIST

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

summary(model, (1, 28, 28))
```


```
------------------------------------------------------------------------------------------
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 10, 24, 24]          260
├─Conv2d: 1-2                            [-1, 20, 8, 8]            5,020
├─Dropout2d: 1-3                         [-1, 20, 8, 8]            --
├─Linear: 1-4                            [-1, 50]                  16,050
├─Linear: 1-5                            [-1, 10]                  510
==========================================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
------------------------------------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.05
Params size (MB): 0.08
Estimated Total Size (MB): 0.14
------------------------------------------------------------------------------------------
```


#### ResNet

```python
import torch
from torchvision import models
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet50()

summary(model, (3, 224, 224))
```


```
------------------------------------------------------------------------------------------
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 64, 112, 112]        9,408
├─BatchNorm2d: 1-2                       [-1, 64, 112, 112]        128
├─ReLU: 1-3                              [-1, 64, 112, 112]        --
├─MaxPool2d: 1-4                         [-1, 64, 56, 56]          --
├─Sequential: 1-5                        [-1, 256, 56, 56]         --
|    └─Bottleneck: 2-1                   [-1, 256, 56, 56]         --
|    |    └─Conv2d: 3-1                  [-1, 64, 56, 56]          4,096
|    |    └─BatchNorm2d: 3-2             [-1, 64, 56, 56]          128
|    |    └─ReLU: 3-3                    [-1, 64, 56, 56]          --
|    |    └─Conv2d: 3-4                  [-1, 64, 56, 56]          36,864
|    |    └─BatchNorm2d: 3-5             [-1, 64, 56, 56]          128
|    |    └─ReLU: 3-6                    [-1, 64, 56, 56]          --
|    |    └─Conv2d: 3-7                  [-1, 256, 56, 56]         16,384
|    |    └─BatchNorm2d: 3-8             [-1, 256, 56, 56]         512
|    |    └─Sequential: 3-9              [-1, 256, 56, 56]         --
|    |    └─ReLU: 3-10                   [-1, 256, 56, 56]         --

  ...
  ...
  ...

├─AdaptiveAvgPool2d: 1-9                 [-1, 2048, 1, 1]          --
├─Linear: 1-10                           [-1, 1000]                2,049,000
==========================================================================================
Total params: 60,192,808
Trainable params: 60,192,808
Non-trainable params: 0
------------------------------------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 344.16
Params size (MB): 229.62
Estimated Total Size (MB): 574.35
------------------------------------------------------------------------------------------


```


#### Multiple Inputs


```python
import torch
import torch.nn as nn
from torchsummary import summary

class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, y):
        x1 = self.features(x)
        x2 = self.features(y)
        return x1, x2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleConv().to(device)

summary(model, [(1, 16, 16), (1, 28, 28)])
```


```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 1, 16, 16]              10
              ReLU-2            [-1, 1, 16, 16]               0
            Conv2d-3            [-1, 1, 28, 28]              10
              ReLU-4            [-1, 1, 28, 28]               0
================================================================
Total params: 20
Trainable params: 20
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.77
Forward/backward pass size (MB): 0.02
Params size (MB): 0.00
Estimated Total Size (MB): 0.78
----------------------------------------------------------------
```

### References
- Thanks to @sksq96, @nmhkahn, and @sangyx for providing the original code this project was based off of.
- For Model Size Estimation @jacobkimmel ([details here](https://github.com/sksq96/pytorch-summary/pull/21))
