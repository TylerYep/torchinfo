# torchinfo
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI version](https://badge.fury.io/py/torch-summary.svg)](https://badge.fury.io/py/torch-summary)
[![Build Status](https://travis-ci.org/TylerYep/torch-summary.svg?branch=master)](https://travis-ci.org/TylerYep/torch-summary)
[![GitHub license](https://img.shields.io/github/license/TylerYep/torch-summary)](https://github.com/TylerYep/torch-summary/blob/master/LICENSE)
[![codecov](https://codecov.io/gh/TylerYep/torch-summary/branch/master/graph/badge.svg)](https://codecov.io/gh/TylerYep/torch-summary)
[![Downloads](https://pepy.tech/badge/torch-summary)](https://pepy.tech/project/torch-summary)

### Announcement: We have moved to `torchinfo`!
`torch-summary` has been renamed to `torchinfo`! Nearly all of the functionality is the same, but the new name will allow us to develop and experiment with additional new features. All links now redirect to `torchinfo`, so please leave an issue there if you have any questions.

The `torch-summary` package will continue to exist for the foreseeable future, so please feel free to pin your desired version (`1.4.3` for Python 3.5, `1.4.4+` for everything else), or try out `torchinfo`. Thanks!

## torch-summary

Torch-summary provides information complementary to what is provided by `print(your_model)` in PyTorch, similar to Tensorflow's `model.summary()` API to view the visualization of the model, which is helpful while debugging your network. In this project, we implement a similar functionality in PyTorch and create a clean, simple interface to use in your projects.

This is a completely rewritten version of the original torchsummary and torchsummaryX projects by @sksq96 and @nmhkahn. This project addresses all of the issues and pull requests left on the original projects by introducing a completely new API.

# Usage
`pip install torch-summary`

# How To Use
```python
from torchsummary import summary

model = ConvNet()
summary(model, (1, 28, 28))
```
```
==========================================================================================
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
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.05
Params size (MB): 0.08
Estimated Total Size (MB): 0.14
==========================================================================================
```

**This version now supports:**
- RNNs, LSTMs, and other recursive layers
- Sequentials & Module Lists
- Branching output used to explore model layers using specified depths
- Returns ModelStatistics object containing all summary data fields
- Configurable columns

**Other new features:**
- Verbose mode to show weights and bias layers
- Accepts either input data or simply the input shape!
- Customizable widths and batch dimension
- Comprehensive unit/output testing, linting, and code coverage testing


# Documentation
```python
"""
Summarize the given PyTorch model. Summarized information includes:
    1) Layer names,
    2) input/output shapes,
    3) kernel shape,
    4) # of parameters,
    5) # of operations (Mult-Adds)

Args:
    model (nn.Module):
            PyTorch model to summarize. The model should be fully in either train()
            or eval() mode. If layers are not all in the same mode, running summary
            may have side effects on batchnorm or dropout statistics. If you
            encounter an issue with this, please open a GitHub issue.

    input_data (Sequence of Sizes or Tensors):
            Example input tensor of the model (dtypes inferred from model input).
            - OR -
            Shape of input data as a List/Tuple/torch.Size
            (dtypes must match model input, default is FloatTensors).
            You should NOT include batch size in the tuple.
            - OR -
            If input_data is not provided, no forward pass through the network is
            performed, and the provided model information is limited to layer names.
            Default: None

    batch_dim (int):
            Batch_dimension of input data. If batch_dim is None, the input data
            is assumed to contain the batch dimension.
            WARNING: in a future version, the default will change to None.
            Default: 0

    branching (bool):
            Whether to use the branching layout for the printed output.
            Default: True

    col_names (Iterable[str]):
            Specify which columns to show in the output. Currently supported:
            ("input_size", "output_size", "num_params", "kernel_size", "mult_adds")
            If input_data is not provided, only "num_params" is used.
            Default: ("output_size", "num_params")

    col_width (int):
            Width of each column.
            Default: 25

    depth (int):
            Number of nested layers to traverse (e.g. Sequentials).
            Default: 3

    device (torch.Device):
            Uses this torch device for model and input_data.
            If not specified, uses result of torch.cuda.is_available().
            Default: None

    dtypes (List[torch.dtype]):
            For multiple inputs, specify the size of both inputs, and
            also specify the types of each parameter here.
            Default: None

    verbose (int):
            0 (quiet): No output
            1 (default): Print model summary
            2 (verbose): Show weight and bias layers in full detail
            Default: 1

    *args, **kwargs:
            Other arguments used in `model.forward` function.

Return:
    ModelStatistics object
            See torchsummary/model_statistics.py for more information.
"""
```

# Examples
## Get Model Summary as String
```python
from torchsummary import summary

model_stats = summary(your_model, (3, 28, 28), verbose=0)
summary_str = str(model_stats)
# summary_str contains the string representation of the summary. See below for examples.
```

## ResNet
```python
import torchvision

model = torchvision.models.resnet50()
summary(model, (3, 224, 224), depth=3)
```
```
==========================================================================================
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
Total mult-adds (G): 11.63
==========================================================================================
Input size (MB): 0.57
Forward/backward pass size (MB): 344.16
Params size (MB): 229.62
Estimated Total Size (MB): 574.35
==========================================================================================
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
    """ Batch-first LSTM model. """
    def __init__(self, vocab_size=20, embed_dim=300, hidden_dim=512, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
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
========================================================================================================================
Layer (type:depth-idx)                   Kernel Shape         Output Shape         Param #              Mult-Adds
========================================================================================================================
Embedding: 1-1                           [300, 20]            [-1, 100, 300]       6,000                6,000
LSTM: 1-2                                --                   [-1, 100, 512]        3,768,320            3,760,128
  weight_ih_l0                           [2048, 300]
  weight_hh_l0                           [2048, 512]
  weight_ih_l1                           [2048, 512]
  weight_hh_l1                           [2048, 512]
Linear: 1-3                              [512, 20]            [-1, 100, 20]        10,260               10,240
========================================================================================================================
Total params: 3,784,580
Trainable params: 3,784,580
Non-trainable params: 0
Total mult-adds (M): 3.78
========================================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 1.03
Params size (MB): 14.44
Estimated Total Size (MB): 15.46
========================================================================================================================
```

## Sequentials & ModuleLists
```python
class ContainerModule(nn.Module):
    """ Model using ModuleList. """

    def __init__(self):
        super().__init__()
        self._layers = nn.ModuleList()
        self._layers.append(nn.Linear(5, 5))
        self._layers.append(ContainerChildModule())
        self._layers.append(nn.Linear(5, 5))

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class ContainerChildModule(nn.Module):
    """ Model using Sequential in different ways. """

    def __init__(self):
        super().__init__()
        self._sequential = nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))
        self._between = nn.Linear(5, 5)

    def forward(self, x):
        out = self._sequential(x)
        out = self._between(out)
        for l in self._sequential:
            out = l(out)

        out = self._sequential(x)
        for l in self._sequential:
            out = l(out)
        return out

summary(ContainerModule(), (5,))
```
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─ModuleList: 1                          []                        --
|    └─Linear: 2-1                       [-1, 5]                   30
|    └─ContainerChildModule: 2-2         [-1, 5]                   --
|    |    └─Sequential: 3-1              [-1, 5]                   --
|    |    |    └─Linear: 4-1             [-1, 5]                   30
|    |    |    └─Linear: 4-2             [-1, 5]                   30
|    |    └─Linear: 3-2                  [-1, 5]                   30
|    |    └─Sequential: 3                []                        --
|    |    |    └─Linear: 4-3             [-1, 5]                   (recursive)
|    |    |    └─Linear: 4-4             [-1, 5]                   (recursive)
|    |    └─Sequential: 3-3              [-1, 5]                   (recursive)
|    |    |    └─Linear: 4-5             [-1, 5]                   (recursive)
|    |    |    └─Linear: 4-6             [-1, 5]                   (recursive)
|    |    |    └─Linear: 4-7             [-1, 5]                   (recursive)
|    |    |    └─Linear: 4-8             [-1, 5]                   (recursive)
|    └─Linear: 2-3                       [-1, 5]                   30
==========================================================================================
Total params: 150
Trainable params: 150
Non-trainable params: 0
Total mult-adds (M): 0.00
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.00
==========================================================================================
```

# Other Examples
```
================================================================
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
================================================================
Input size (MB): 0.77
Forward/backward pass size (MB): 0.02
Params size (MB): 0.00
Estimated Total Size (MB): 0.78
================================================================
```

# Future Plans
- Change project name to `torchinfo`. The API will eventually mature to the point that this project deserves its own name.
- Support all types of inputs - showing tuples and dict inputs cleanly rather than only using the first tensor in the list.
- Default `batch_dim` to `None` rather than `0`. Users must specify the batch size in the input shape, or pass in `batch_dim=0` in order to ignore it.
- FunctionalNet unused; figure out a way to hook into functional layers.

# Contributing
All issues and pull requests are much appreciated! If you are wondering how to build the project:

- torch-summary is actively developed using the lastest version of Python.
    - Changes should be backward compatible with Python 3.6, but this is subject to change in the future.
    - Run `pip install -r requirements-dev.txt`. We use the latest versions of all dev packages.
    - First, be sure to run `./scripts/install-hooks`
    - To run all tests and use auto-formatting tools, check out `scripts/run-tests`.
    - To only run unit tests, run `pytest`.

# References
- Thanks to @sksq96, @nmhkahn, and @sangyx for providing the original code this project was based off of.
- For Model Size Estimation @jacobkimmel ([details here](https://github.com/sksq96/pytorch-summary/pull/21))
