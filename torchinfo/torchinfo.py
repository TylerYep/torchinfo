""" torchinfo.py """
import sys
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from torch.jit import ScriptModule
from torch.utils.hooks import RemovableHandle

from .formatting import (
    ALL_COLUMN_SETTINGS,
    ALL_ROW_SETTINGS,
    FormattingOptions,
    Verbosity,
)
from .layer_info import LayerInfo
from .model_statistics import CORRECTED_INPUT_SIZE_TYPE, ModelStatistics

# Some modules do the computation themselves using parameters
# or the parameters of children. Treat these as layers.
LAYER_MODULES = (torch.nn.MultiheadAttention,)
INPUT_SIZE_TYPE = Sequence[Union[int, Sequence[Any], torch.Size]]
INPUT_DATA_TYPE = Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]]
DEFAULT_COLUMN_NAMES = ("output_size", "num_params")
DEFAULT_ROW_SETTINGS = ("depth",)


def summary(
    model: nn.Module,
    input_size: Optional[INPUT_SIZE_TYPE] = None,
    input_data: Optional[INPUT_DATA_TYPE] = None,
    batch_dim: Optional[int] = None,
    col_names: Optional[Iterable[str]] = None,
    col_width: int = 25,
    depth: int = 3,
    device: Optional[Union[torch.device, str]] = None,
    dtypes: Optional[List[torch.dtype]] = None,
    row_settings: Optional[Iterable[str]] = None,
    verbose: Optional[int] = None,
    **kwargs: Any,
) -> ModelStatistics:
    """
    Summarize the given PyTorch model. Summarized information includes:
        1) Layer names,
        2) input/output shapes,
        3) kernel shape,
        4) # of parameters,
        5) # of operations (Mult-Adds)

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
                Example input tensor of the model (dtypes inferred from model input).
                Default: None

        batch_dim (int):
                Batch_dimension of input data. If batch_dim is None, assume
                input_data / input_size contains the batch dimension, which is used
                in all calculations. Else, expand all tensors to contain the batch_dim.
                Specifying batch_dim can be an runtime optimization, since if batch_dim
                is specified, torchinfo uses a batch size of 1 for the forward pass.
                Default: None

        col_names (Iterable[str]):
                Specify which columns to show in the output. Currently supported: (
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
                    "mult_adds",
                )
                Default: ("output_size", "num_params")
                If input_data / input_size are not provided, only "num_params" is used.

        col_width (int):
                Width of each column.
                Default: 25

        depth (int):
                Number of nested layers to display (e.g. Sequentials).
                Default: 3

        device (torch.Device):
                Uses this torch device for model and input_data.
                If not specified, uses result of torch.cuda.is_available().
                Default: None

        dtypes (List[torch.dtype]):
                For multiple inputs, specify the size of both inputs, and
                also specify the types of each parameter here.
                Default: None

        row_settings (Iterable[str]):
                Specify which features to show in a row. Currently supported: (
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
    input_data_specified = input_data is not None or input_size is not None

    if col_names is None:
        col_names = DEFAULT_COLUMN_NAMES if input_data_specified else ("num_params",)

    if row_settings is None:
        row_settings = DEFAULT_ROW_SETTINGS

    if verbose is None:
        # pylint: disable=no-member
        verbose = 0 if hasattr(sys, "ps1") and sys.ps1 else 1

    validate_user_params(
        input_data, input_size, col_names, col_width, row_settings, verbose
    )
    summary_list: List[LayerInfo] = []
    hooks: Optional[List[RemovableHandle]] = [] if input_data_specified else None
    idx: Dict[int, int] = {}
    named_module = (model.__class__.__name__, model)
    apply_hooks(named_module, model, batch_dim, summary_list, idx, hooks)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct_input_size: CORRECTED_INPUT_SIZE_TYPE = []
    if input_data is not None:
        x, correct_input_size = process_input_data(input_data, device)

    if input_size is not None:
        if dtypes is None:
            dtypes = [torch.float] * len(input_size)
        correct_input_size = get_correct_input_sizes(input_size)
        x = get_input_tensor(correct_input_size, batch_dim, dtypes, device)

    if input_data_specified:
        kwargs = set_device(kwargs, device)
        saved_model_mode = model.training
        try:
            model.eval()
            with torch.no_grad():
                if isinstance(x, (list, tuple)):
                    _ = model.to(device)(*x, **kwargs)
                elif isinstance(x, dict):
                    _ = model.to(device)(**x, **kwargs)
                else:
                    # Should not reach this point, since process_input_data ensures
                    # x is either a list, tuple, or dict
                    raise ValueError("Unknown input type")
        except Exception as e:
            executed_layers = [layer for layer in summary_list if layer.executed]
            raise RuntimeError(
                "Failed to run torchinfo. See above stack traces for more details. "
                f"Executed layers up to: {executed_layers}"
            ) from e
        finally:
            if hooks is not None:
                for hook in hooks:
                    hook.remove()
            model.train(saved_model_mode)

    if not summary_list or summary_list[0].var_name != model.__class__.__name__:
        summary_list.insert(0, LayerInfo("", model, 0))

    formatting = FormattingOptions(depth, verbose, col_names, col_width, row_settings)
    results = ModelStatistics(summary_list, correct_input_size, formatting)
    if verbose > Verbosity.QUIET.value:
        print(results)
    return results


def validate_user_params(
    input_data: Optional[INPUT_DATA_TYPE],
    input_size: Optional[INPUT_SIZE_TYPE],
    col_names: Iterable[str],
    col_width: int,
    row_settings: Iterable[str],
    verbose: int,
) -> None:
    """Raise exceptions if the user's input is invalid."""
    if col_width <= 0:
        raise ValueError(f"Column width must be greater than 0: col_width={col_width}")
    if verbose not in (0, 1, 2):
        raise ValueError(
            "Verbose must be either 0 (quiet), 1 (default), or 2 (verbose)."
        )
    both_input_specified = input_data is not None and input_size is not None
    if both_input_specified:
        raise RuntimeError("Only one of (input_data, input_size) should be specified.")

    neither_input_specified = input_data is None and input_size is None
    for col in col_names:
        if col not in ALL_COLUMN_SETTINGS:
            raise ValueError(f"Column {col} is not a valid column name.")
        if neither_input_specified and col not in ("num_params", "kernel_size"):
            raise ValueError(
                f"You must pass input_data or input_size in order to use column {col}"
            )
    for setting in row_settings:
        if setting not in ALL_ROW_SETTINGS:
            raise ValueError(f"Row setting {setting} is not a valid setting.")


def set_device(data: Any, device: Union[torch.device, str]) -> Any:
    """Sets device for all input types and collections of input types."""
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)

    # Recursively apply to collection items
    elem_type = type(data)
    if isinstance(data, Mapping):
        return elem_type({k: set_device(v, device) for k, v in data.items()})
    if isinstance(data, tuple) and hasattr(data, "_fields"):  # Named tuple
        return elem_type(*(set_device(d, device) for d in data))
    if isinstance(data, Iterable) and not isinstance(data, str):
        return elem_type([set_device(d, device) for d in data])
    # Data is neither a tensor nor a collection
    return data


# TODO: generalize to all input types
# recurse like in set_device to locate all tensors, and report back all sizes.
def process_input_data(
    input_data: INPUT_DATA_TYPE, device: Union[torch.device, str]
) -> Tuple[INPUT_DATA_TYPE, CORRECTED_INPUT_SIZE_TYPE]:
    """Reads sample input data to get the input size."""
    x = None
    if isinstance(input_data, torch.Tensor):
        input_size = get_correct_input_sizes(input_data.size())
        x = [set_device(input_data, device)]

    elif isinstance(input_data, (list, tuple)) and all(
        isinstance(data, torch.Tensor) for data in input_data
    ):
        input_sizes = [data.size() for data in input_data]
        input_size = get_correct_input_sizes(input_sizes)
        x = set_device(input_data, device)

    elif isinstance(input_data, dict) and all(
        isinstance(data, torch.Tensor) for data in input_data.values()
    ):
        input_sizes = [data.size() for data in input_data.values()]
        input_size = get_correct_input_sizes(input_sizes)
        x = set_device(input_data, device)

    if x is None:
        raise RuntimeError(
            "Input type is not recognized. Please ensure input_data is valid.\n"
            "For multiple inputs to the network, ensure input_data passed in is "
            "a sequence of tensors or a list of tuple sizes. If you are having "
            "trouble here, please submit a GitHub issue."
        )

    return x, input_size


def get_input_tensor(
    input_size: CORRECTED_INPUT_SIZE_TYPE,
    batch_dim: Optional[int],
    dtypes: List[torch.dtype],
    device: Union[torch.device, str],
) -> List[torch.Tensor]:
    """Get input_tensor with batch size 1 for use in model.forward()"""
    x = []
    for size, dtype in zip(input_size, dtypes):
        input_tensor = torch.rand(*size)
        if batch_dim is not None:
            input_tensor = input_tensor.unsqueeze(dim=batch_dim)
        x.append(input_tensor.to(device).type(dtype))
    return x


def flatten(nested_array: INPUT_SIZE_TYPE) -> Iterator[Any]:
    """Flattens a nested array."""
    for item in nested_array:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item


def get_correct_input_sizes(input_size: INPUT_SIZE_TYPE) -> CORRECTED_INPUT_SIZE_TYPE:
    """
    Convert input_size to the correct form, which is a list of tuples.
    Also handles multiple inputs to the network.
    """
    if not isinstance(input_size, (list, tuple)):
        raise TypeError(
            "Input_size is not a recognized type. Please ensure input_size is valid.\n"
            "For multiple inputs to the network, ensure input_size is a list of tuple "
            "sizes. If you are having trouble here, please submit a GitHub issue."
        )
    if not input_size or any(size <= 0 for size in flatten(input_size)):
        raise ValueError("Input_data is invalid, or negative size found in input_data.")

    if isinstance(input_size, list) and isinstance(input_size[0], int):
        return [tuple(input_size)]
    if isinstance(input_size, list):
        return input_size
    if isinstance(input_size, tuple) and isinstance(input_size[0], tuple):
        return list(input_size)
    return [input_size]


def apply_hooks(
    named_module: Tuple[str, nn.Module],
    orig_model: nn.Module,
    batch_dim: Optional[int],
    summary_list: List[LayerInfo],
    idx: Dict[int, int],
    hooks: Optional[List[RemovableHandle]],
    curr_depth: int = 0,
    parent_info: Optional[LayerInfo] = None,
) -> None:
    """
    If input_data is provided, recursively adds hooks to all layers of the model.
    Else, fills summary_list with layer info without computing a
    forward pass through the network.
    """
    # Fallback is used if the layer's pre-hook is never called, for example in
    # ModuleLists or Sequentials.
    var_name, module = named_module
    info = LayerInfo(var_name, module, curr_depth, None, parent_info)

    def pre_hook(module: nn.Module, inputs: Any) -> None:
        """Create a LayerInfo object to aggregate information about that layer."""
        del inputs
        nonlocal info
        idx[curr_depth] = idx.get(curr_depth, 0) + 1
        info = LayerInfo(var_name, module, curr_depth, idx[curr_depth], parent_info)
        info.calculate_num_params()
        info.check_recursive(summary_list)
        summary_list.append(info)

    def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
        """Update LayerInfo after forward pass."""
        del module
        info.input_size = info.calculate_size(inputs, batch_dim)
        info.output_size = info.calculate_size(outputs, batch_dim)
        info.calculate_macs()
        info.executed = True

    submodules = [m for m in module.modules() if m is not orig_model]
    if module != orig_model or isinstance(module, LAYER_MODULES) or not submodules:
        if hooks is None or isinstance(module, (nn.ParameterList, ScriptModule)):
            pre_hook(module, None)
        else:
            hooks.append(module.register_forward_pre_hook(pre_hook))
            hooks.append(module.register_forward_hook(hook))

    for child in module.named_children():
        apply_hooks(
            child, orig_model, batch_dim, summary_list, idx, hooks, curr_depth + 1, info
        )
