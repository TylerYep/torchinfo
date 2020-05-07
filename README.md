# torch-summary
[![PyPI version](https://badge.fury.io/py/torch-summary.svg)](https://badge.fury.io/py/torch-summary)

Keras has a neat API to view the visualization of the model which is very helpful while debugging your network. In this project, we attempt to do the same in PyTorch. The goal is to provide information complementary to what is provided by `print(your_model)` in PyTorch.

This is a rewritten version of the original torchsummary and torchsummaryX projects by @sksq96 and @nmhkahn.
There are quite a few pull requests on the original project (which hasn't been updated in over a year), so I decided to take a stab at improving and consolidating some of the features.

**This version now supports:**
- RNNs, LSTMs, and other recursive layers
- Branching output to explore model layers using specified depths
- Returns ModelStatistics object to access summary data
- Configurable columns of returned data

**Other features:**
- Verbose mode to show specific weights and bias layers
- Accepts either input data or simply the input shape to work!
- Customizable widths and custom batch dimension.
- More comprehensive testing using pytest


# Usage
`pip install torch-summary`

or

`git clone https://github.com/tyleryep/torch-summary.git`


```python
from torchsummary import summary
summary(your_model, input_data=(C, H, W))
```

# Documentation
```python
"""
Summarize the given PyTorch model. Summarized information includes:
    1) output shape,
    2) kernel shape,
    3) number of the parameters
    4) operations (Mult-Adds)
Args:
    model (Module): Model to summarize
    input_data (Sequence of Sizes or Tensors):
        Example input tensor of the model (dtypes inferred from model input).
        - OR -
        Shape of input data as a List/Tuple/torch.Size (dtypes must match model input,
        default is FloatTensors). NOTE: For scalar parameters, use torch.Size([]).
    branching (bool): Whether to use the branching layout for the printed output.
    depth (int): number of nested layers to traverse (e.g. Sequentials)
    verbose (int):
        0 (quiet): No output
        1 (default): Print model summary
        2 (verbose): Show weight and bias layers in full detail
    col_names (List): specify which columns to show in the output. Currently supported:
        ['output_size', 'num_params', 'kernel_size', 'mult_adds']
    col_width (int): width of each column
    dtypes (List or None): for multiple inputs or args, must specify the size of both inputs.
        You must also specify the types of each parameter here.
    batch_dim (int): batch_dimension of input data
    device (torch.Device): If specified, uses this torch device for the model and model's input.
        Else defaults to torch.cuda.is_available().
    args, kwargs: Other arguments used in `model.forward` function.
"""
```


# Examples
## Get Model Summary as String
```python
from torchsummary import summary

model_stats = summary(your_model, input_data=(C, H, W), verbose=0)
summary_str = str(model_stats)
```

## CNN for MNIST

```python
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


model = CNN()
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


## Multiple Inputs w/ Different Data Types

```python
class MultipleInputNetDifferentDtypes(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)

        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = x2.type(torch.float)
        x2 = F.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = torch.cat((x1, x2), 0)
        return F.log_softmax(x, dim=1)


summary(model, [(1, 300), (1, 300)], dtypes=[torch.float, torch.long])
```
Alternatively, you can also pass in the input_data itself, and
torchsummary will automatically infer the data types.

```python
input_data = torch.randn(1, 300)
other_input_data = torch.randn(1, 300).long()
model = MultipleInputNetDifferentDtypes()

summary(model, input_data, other_input_data, ...)
```

## Explore Different Configurations
```python
class LSTMNet(nn.Module):
    def __init__(self, vocab_size=20, embed_dim=300, hidden_dim=512, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        out, hidden = self.encoder(embed)
        out = self.decoder(out)
        out = out.view(-1, out.size(2))
        return out, hidden

summary(
    LSTMNet(),
    (100,),
    dtypes=[torch.long],
    branching=False,
    verbose=2,
    col_width=16,
    col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
)
```

```
--------------------------------------------------------------------------------------------------------
Layer (type:depth-idx)         Kernel Shape         Output Shape         Param #          Mult-Adds
========================================================================================================
Embedding: 1-1                 [300, 20]            [-1, 100, 300]       6,000            6,000
LSTM: 1-2                       --                  [2, 100, 512]        3,768,320        3,760,128
  weight_ih_l0                 [2048, 300]
  weight_hh_l0                 [2048, 512]
  weight_ih_l1                 [2048, 512]
  weight_hh_l1                 [2048, 512]
Linear: 1-3                    [512, 20]            [-1, 100, 20]        10,260           10,240
========================================================================================================
Total params: 3,784,580
Trainable params: 3,784,580
Non-trainable params: 0
--------------------------------------------------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.03
Params size (MB): 14.44
Estimated Total Size (MB): 15.46
--------------------------------------------------------------------------------------------------------
```


## ResNet

```python
import torchvision

model = torchvision.models.resnet50()
summary(model, (3, 224, 224), depth=3)
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


# Other Examples
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

# References
- Thanks to @sksq96, @nmhkahn, and @sangyx for providing the original code this project was based off of.
- For Model Size Estimation @jacobkimmel ([details here](https://github.com/sksq96/pytorch-summary/pull/21))
