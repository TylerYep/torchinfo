# torchinfo

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI version](https://badge.fury.io/py/torchinfo.svg)](https://badge.fury.io/py/torchinfo)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/torchinfo)](https://anaconda.org/conda-forge/torchinfo)
[![Build Status](https://github.com/TylerYep/torchinfo/actions/workflows/test.yml/badge.svg)](https://github.com/TylerYep/torchinfo/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/TylerYep/torchinfo/main.svg)](https://results.pre-commit.ci/latest/github/TylerYep/torchinfo/main)
[![GitHub license](https://img.shields.io/github/license/TylerYep/torchinfo)](https://github.com/TylerYep/torchinfo/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/TylerYep/torchinfo/branch/main/graph/badge.svg)](https://codecov.io/gh/TylerYep/torchinfo)
[![Downloads](https://pepy.tech/badge/torchinfo)](https://pepy.tech/project/torchinfo)

(formerly torch-summary)

Torchinfo provides information complementary to what is provided by `print(your_model)` in PyTorch, similar to Tensorflow's `model.summary()` API to view the visualization of the model, which is helpful while debugging your network. In this project, we implement a similar functionality in PyTorch and create a clean, simple interface to use in your projects.

This is a completely rewritten version of the original torchsummary and torchsummaryX projects by @sksq96 and @nmhkahn. This project addresses all of the issues and pull requests left on the original projects by introducing a completely new API.

Supports PyTorch versions 1.4.0+.

# Usage

```
pip install torchinfo
```

Alternatively, via conda:

```
conda install -c conda-forge torchinfo
```

# How To Use

```python
from torchinfo import summary

model = ConvNet()
batch_size = 16
summary(model, input_size=(batch_size, 1, 28, 28))
```

```
================================================================================================================
Layer (type:depth-idx)          Input Shape          Output Shape         Param #            Mult-Adds
================================================================================================================
SingleInputNet                  --                   --                   --                  --
├─Conv2d: 1-1                   [7, 1, 28, 28]       [7, 10, 24, 24]      260                1,048,320
├─Conv2d: 1-2                   [7, 10, 12, 12]      [7, 20, 8, 8]        5,020              2,248,960
├─Dropout2d: 1-3                [7, 20, 8, 8]        [7, 20, 8, 8]        --                 --
├─Linear: 1-4                   [7, 320]             [7, 50]              16,050             112,350
├─Linear: 1-5                   [7, 50]              [7, 10]              510                3,570
================================================================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
Total mult-adds (M): 3.41
================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 0.40
Params size (MB): 0.09
Estimated Total Size (MB): 0.51
================================================================================================================
```

<!-- single_input_all_cols.out -->

Note: if you are using a Jupyter Notebook or Google Colab, `summary(model, ...)` must be the returned value of the cell.
If it is not, you should wrap the summary in a print(), e.g. `print(summary(model, ...))`.
See `tests/jupyter_test.ipynb` for examples.

**This version now supports:**

- RNNs, LSTMs, and other recursive layers
- Branching output used to explore model layers using specified depths
- Returns ModelStatistics object containing all summary data fields
- Configurable rows/columns
- Jupyter Notebook / Google Colab

**Other new features:**

- Verbose mode to show weights and bias layers
- Accepts either input data or simply the input shape!
- Customizable line widths and batch dimension
- Comprehensive unit/output testing, linting, and code coverage testing

**Community Contributions:**

- Sequentials & ModuleLists (thanks to @roym899)
- Improved Mult-Add calculations (thanks to @TE-StefanUhlich, @zmzhang2000)
- Dict/Misc input data (thanks to @e-dorigatti)
- Pruned layer support (thanks to @MajorCarrot)

# Documentation

```python
def summary(
    model: nn.Module,
    input_size: Optional[INPUT_SIZE_TYPE] = None,
    input_data: Optional[INPUT_DATA_TYPE] = None,
    batch_dim: Optional[int] = None,
    cache_forward_pass: Optional[bool] = None,
    col_names: Optional[Iterable[str]] = None,
    col_width: int = 25,
    depth: int = 3,
    device: Optional[torch.device] = None,
    dtypes: Optional[List[torch.dtype]] = None,
    mode: str | None = None,
    row_settings: Optional[Iterable[str]] = None,
    verbose: int = 1,
    **kwargs: Any,
) -> ModelStatistics:
"""
Summarize the given PyTorch model. Summarized information includes:
    1) Layer names,
    2) input/output shapes,
    3) kernel shape,
    4) # of parameters,
    5) # of operations (Mult-Adds),
    6) whether layer is trainable

NOTE: If neither input_data or input_size are provided, no forward pass through the
network is performed, and the provided model information is limited to layer names.

Args:
    model (nn.Module):
            PyTorch model to summarize. The model should be fully in either train()
            or eval() mode. If layers are not all in the same mode, running summary
            may have side effects on batchnorm or dropout statistics. If you
            encounter an issue with this, please open a GitHub issue.

    input_size (Sequence of Sizes):
            Shape of input data as a List/Tuple/torch.Size
            (dtypes must match model input, default is FloatTensors).
            You should include batch size in the tuple.
            Default: None

    input_data (Sequence of Tensors):
            Arguments for the model's forward pass (dtypes inferred).
            If the forward() function takes several parameters, pass in a list of
            args or a dict of kwargs (if your forward() function takes in a dict
            as its only argument, wrap it in a list).
            Default: None

    batch_dim (int):
            Batch_dimension of input data. If batch_dim is None, assume
            input_data / input_size contains the batch dimension, which is used
            in all calculations. Else, expand all tensors to contain the batch_dim.
            Specifying batch_dim can be an runtime optimization, since if batch_dim
            is specified, torchinfo uses a batch size of 1 for the forward pass.
            Default: None

    cache_forward_pass (bool):
            If True, cache the run of the forward() function using the model
            class name as the key. If the forward pass is an expensive operation,
            this can make it easier to modify the formatting of your model
            summary, e.g. changing the depth or enabled column types, especially
            in Jupyter Notebooks.
            WARNING: Modifying the model architecture or input data/input size when
            this feature is enabled does not invalidate the cache or re-run the
            forward pass, and can cause incorrect summaries as a result.
            Default: False

    col_names (Iterable[str]):
            Specify which columns to show in the output. Currently supported: (
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
                "mult_adds",
                "trainable",
            )
            Default: ("output_size", "num_params")
            If input_data / input_size are not provided, only "num_params" is used.

    col_width (int):
            Width of each column.
            Default: 25

    depth (int):
            Depth of nested layers to display (e.g. Sequentials).
            Nested layers below this depth will not be displayed in the summary.
            Default: 3

    device (torch.Device):
            Uses this torch device for model and input_data.
            If not specified, uses result of torch.cuda.is_available().
            Default: None

    dtypes (List[torch.dtype]):
            If you use input_size, torchinfo assumes your input uses FloatTensors.
            If your model use a different data type, specify that dtype.
            For multiple inputs, specify the size of both inputs, and
            also specify the types of each parameter here.
            Default: None

    mode (str)
            Either "train" or "eval", which determines whether we call
            model.train() or model.eval() before calling summary().
            Default: "eval".

    row_settings (Iterable[str]):
            Specify which features to show in a row. Currently supported: (
                "ascii_only",
                "depth",
                "var_names",
            )
            Default: ("depth",)

    verbose (int):
            0 (quiet): No output
            1 (default): Print model summary
            2 (verbose): Show weight and bias layers in full detail
            Default: 1
            If using a Juypter Notebook or Google Colab, the default is 0.

    **kwargs:
            Other arguments used in `model.forward` function. Passing *args is no
            longer supported.

Return:
    ModelStatistics object
            See torchinfo/model_statistics.py for more information.
"""
```

# Examples

## Get Model Summary as String

```python
from torchinfo import summary

model_stats = summary(your_model, (1, 3, 28, 28), verbose=0)
summary_str = str(model_stats)
# summary_str contains the string representation of the summary!
```

## Explore Different Configurations

```python
class LSTMNet(nn.Module):
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
    (1, 100),
    dtypes=[torch.long],
    verbose=2,
    col_width=16,
    col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
    row_settings=["var_names"],
)
```

```
========================================================================================================================
Layer (type (var_name))                  Kernel Shape         Output Shape         Param #              Mult-Adds
========================================================================================================================
LSTMNet                                  --                   --                   --                   --
├─Embedding (embedding)                  [300, 20]            [1, 100, 300]        6,000                6,000
│    └─weight                            [300, 20]                                 └─6,000
├─LSTM (encoder)                         --                   [1, 100, 512]        3,768,320            376,832,000
│    └─weight_ih_l0                      [2048, 300]                               ├─614,400
│    └─weight_hh_l0                      [2048, 512]                               ├─1,048,576
│    └─bias_ih_l0                        [2048]                                    ├─2,048
│    └─bias_hh_l0                        [2048]                                    ├─2,048
│    └─weight_ih_l1                      [2048, 512]                               ├─1,048,576
│    └─weight_hh_l1                      [2048, 512]                               ├─1,048,576
│    └─bias_ih_l1                        [2048]                                    ├─2,048
│    └─bias_hh_l1                        [2048]                                    └─2,048
├─Linear (decoder)                       [512, 20]            [1, 100, 20]         10,260               10,260
│    └─weight                            [512, 20]                                 ├─10,240
│    └─bias                              [20]                                      └─20
========================================================================================================================
Total params: 3,784,580
Trainable params: 3,784,580
Non-trainable params: 0
Total mult-adds (M): 376.85
========================================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 15.14
Estimated Total Size (MB): 15.80
========================================================================================================================

```

<!-- lstm.out -->

## ResNet

```python
import torchvision

model = torchvision.models.resnet152()
summary(model, (1, 3, 224, 224), depth=3)
```

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ResNet                                   --                        --
├─Conv2d: 1-1                            [1, 64, 112, 112]         9,408
├─BatchNorm2d: 1-2                       [1, 64, 112, 112]         128
├─ReLU: 1-3                              [1, 64, 112, 112]         --
├─MaxPool2d: 1-4                         [1, 64, 56, 56]           --
├─Sequential: 1-5                        [1, 256, 56, 56]          --
│    └─Bottleneck: 2-1                   [1, 256, 56, 56]          --
│    │    └─Conv2d: 3-1                  [1, 64, 56, 56]           4,096
│    │    └─BatchNorm2d: 3-2             [1, 64, 56, 56]           128
│    │    └─ReLU: 3-3                    [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-4                  [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-5             [1, 64, 56, 56]           128
│    │    └─ReLU: 3-6                    [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-7                  [1, 256, 56, 56]          16,384
│    │    └─BatchNorm2d: 3-8             [1, 256, 56, 56]          512
│    │    └─Sequential: 3-9              [1, 256, 56, 56]          16,896
│    │    └─ReLU: 3-10                   [1, 256, 56, 56]          --
│    └─Bottleneck: 2-2                   [1, 256, 56, 56]          --

  ...
  ...
  ...

├─AdaptiveAvgPool2d: 1-9                 [1, 2048, 1, 1]           --
├─Linear: 1-10                           [1, 1000]                 2,049,000
==========================================================================================
Total params: 60,192,808
Trainable params: 60,192,808
Non-trainable params: 0
Total mult-adds (G): 11.51
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 360.87
Params size (MB): 240.77
Estimated Total Size (MB): 602.25
==========================================================================================
```

<!-- resnet152.out -->

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
torchinfo will automatically infer the data types.

```python
input_data = torch.randn(1, 300)
other_input_data = torch.randn(1, 300).long()
model = MultipleInputNetDifferentDtypes()

summary(model, input_data=[input_data, other_input_data, ...])
```

## Sequentials & ModuleLists

```python
class ContainerModule(nn.Module):

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

summary(ContainerModule(), (1, 5))
```

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ContainerModule                          --                        --
├─ModuleList: 1-1                        --                        --
│    └─Linear: 2-1                       [1, 5]                    30
│    └─ContainerChildModule: 2-2         [1, 5]                    --
│    │    └─Sequential: 3-1              [1, 5]                    --
│    │    │    └─Linear: 4-1             [1, 5]                    30
│    │    │    └─Linear: 4-2             [1, 5]                    30
│    │    └─Linear: 3-2                  [1, 5]                    30
│    │    └─Sequential: 3                --                        --
│    │    │    └─Linear: 4-3             [1, 5]                    (recursive)
│    │    │    └─Linear: 4-4             [1, 5]                    (recursive)
│    │    └─Sequential: 3-3              [1, 5]                    (recursive)
│    │    │    └─Linear: 4-5             [1, 5]                    (recursive)
│    │    │    └─Linear: 4-6             [1, 5]                    (recursive)
│    │    │    └─Linear: 4-7             [1, 5]                    (recursive)
│    │    │    └─Linear: 4-8             [1, 5]                    (recursive)
│    └─Linear: 2-3                       [1, 5]                    30
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

<!-- container.out -->

# Contributing

All issues and pull requests are much appreciated! If you are wondering how to build the project:

- torchinfo is actively developed using the lastest version of Python.
  - Changes should be backward compatible to Python 3.7, and will follow Python's End-of-Life guidance for old versions.
  - Run `pip install -r requirements-dev.txt`. We use the latest versions of all dev packages.
  - Run `pre-commit install`.
  - To use auto-formatting tools, use `pre-commit run -a`.
  - To run unit tests, run `pytest`.
  - To update the expected output files, run `pytest --overwrite`.
  - To skip output file tests, use `pytest --no-output`

# References

- Thanks to @sksq96, @nmhkahn, and @sangyx for providing the inspiration for this project.
- For Model Size Estimation @jacobkimmel ([details here](https://github.com/sksq96/pytorch-summary/pull/21))
