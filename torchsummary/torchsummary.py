""" torchsummary.py """
from typing import Any, Dict, Generator, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .formatting import FormattingOptions, Verbosity
from .layer_info import LayerInfo
from .model_statistics import CORRECTED_INPUT_SIZE_TYPE, ModelStatistics

# Some modules do the computation themselves using parameters
# or the parameters of children. Treat these as layers.
LAYER_MODULES = (torch.nn.MultiheadAttention,)
INPUT_SIZE_TYPE = Sequence[Union[int, Sequence[Any], torch.Size]]
INPUT_DATA_TYPE = Optional[Union[torch.Tensor, torch.Size, Sequence[torch.Tensor], INPUT_SIZE_TYPE]]


def summary(
    model: nn.Module,
    input_data: INPUT_DATA_TYPE = None,
    *args: Any,
    batch_dim: int = 0,
    branching: bool = True,
    col_names: Sequence[str] = ("output_size", "num_params"),
    col_width: int = 25,
    depth: int = 3,
    device: Optional[torch.device] = None,
    dtypes: Optional[List[torch.dtype]] = None,
    verbose: int = 1,
    **kwargs: Any
) -> ModelStatistics:
    """
    Summarize the given PyTorch model. Summarized information includes:
        1) Layer names,
        2) input/output shapes,
        3) kernel shape,
        4) # of parameters,
        5) # of operations (Mult-Adds)

    Args:
        model (nn.Module):
                PyTorch model to summarize

        input_data (Sequence of Sizes or Tensors):
                Example input tensor of the model (dtypes inferred from model input).
                - OR -
                Shape of input data as a List/Tuple/torch.Size (dtypes must match model input,
                default is FloatTensors). Should NOT include batch size in the tuple.
                - OR -
                If input_data is not provided, no forward pass through the network is performed,
                and the provided model information is limited to layer names.

        batch_dim (int):
                Batch_dimension of input data. Default: 0

        branching (bool):
                Whether to use the branching layout for the printed output. Default: True

        col_names (Sequence[str]):
                Specify which columns to show in the output. Currently supported:
                        ("input_size", "output_size", "num_params", "kernel_size", "mult_adds")
                Default: ("output_size", "num_params")

        col_width (int):
                Width of each column. Default: 25

        depth (int):
                Number of nested layers to traverse (e.g. Sequentials). Default: 3

        device (torch.Device):
                Uses this torch device for model and input_data.
                If not specified, uses result of torch.cuda.is_available(). Default: None

        dtypes (List[torch.dtype]):
                For multiple inputs, specify the size of both inputs, and
                also specify the types of each parameter here. Default: None

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
    assert verbose in (0, 1, 2)
    input_size = []  # type: CORRECTED_INPUT_SIZE_TYPE
    summary_list = []  # type: List[LayerInfo]
    hooks = None if input_data is None else []  # type: Optional[List[RemovableHandle]]
    idx = {}  # type: Dict[int, int]
    apply_hooks(model, model, batch_dim, depth, summary_list, idx, hooks)

    if input_data is not None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x, input_size = process_input_data(input_data, batch_dim, device, dtypes)
        args, kwargs = set_device(args, device), set_device(kwargs, device)
        try:
            with torch.no_grad():
                _ = model.to(device)(*x, *args, **kwargs)  # type: ignore
        except Exception:
            executed_layers = [layer for layer in summary_list if layer.executed]
            print("Failed to run torchsummary, executed layers up to: {}".format(executed_layers))
            raise
        finally:
            if hooks is not None:
                for hook in hooks:
                    hook.remove()

    formatting = FormattingOptions(branching, depth, verbose, col_names, col_width)
    formatting.set_layer_name_width(summary_list)
    results = ModelStatistics(summary_list, input_size, formatting)
    if verbose > Verbosity.QUIET.value:
        print(results)
    return results


def set_device(data: Any, device: torch.device) -> Any:
    """ Sets device for all input types and collections of input types. """
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)

    # Recursively apply to collection items
    elem_type = type(data)
    if isinstance(data, Mapping):
        return elem_type({k: set_device(v, device) for k, v in data.items()})
    if isinstance(data, tuple) and hasattr(data, "_fields"):  # Named tuple
        return elem_type(*(set_device(d, device) for d in data))
    if isinstance(data, Sequence) and not isinstance(data, str):
        return elem_type([set_device(d, device) for d in data])
    # Data is neither a tensor nor a collection
    return data


def process_input_data(
    input_data: INPUT_DATA_TYPE,
    batch_dim: int,
    device: torch.device,
    dtypes: Optional[List[torch.dtype]],
) -> Tuple[INPUT_DATA_TYPE, CORRECTED_INPUT_SIZE_TYPE]:
    """ Create sample input data and the corrected input size. """
    if isinstance(input_data, torch.Tensor):
        input_size = get_correct_input_sizes(input_data.size())
        x = [input_data.to(device)]

    elif isinstance(input_data, (list, tuple)):
        if all(isinstance(data, torch.Tensor) for data in input_data):
            input_sizes = [data.size() for data in input_data]  # type: ignore
            input_size = get_correct_input_sizes(input_sizes)
            x = set_device(input_data, device)
        else:
            if dtypes is None:
                dtypes = [torch.float] * len(input_data)
            input_size = get_correct_input_sizes(input_data)
            x = get_input_tensor(input_size, batch_dim, dtypes, device)

    else:
        raise TypeError(
            "Input type is not recognized. Please ensure input_data is valid.\n"
            "For multiple inputs to the network, ensure input_data passed in is "
            "a sequence of tensors or a list of tuple sizes. If you are having trouble here, "
            "please submit a GitHub issue."
        )

    return x, input_size


def get_input_tensor(
    input_size: CORRECTED_INPUT_SIZE_TYPE,
    batch_dim: int,
    dtypes: List[torch.dtype],
    device: torch.device,
) -> List[torch.Tensor]:
    """ Get input_tensor with batch size 2 for use in model.forward() """
    x = []
    for size, dtype in zip(input_size, dtypes):
        # add batch_size of 2 for BatchNorm
        if isinstance(size, (list, tuple)):
            # Case: input_tensor is a list of dimensions
            input_tensor = torch.rand(*size)
            input_tensor = input_tensor.unsqueeze(dim=batch_dim)
            input_tensor = torch.cat([input_tensor] * 2, dim=batch_dim)
        result = input_tensor.to(device).type(dtype)
        if isinstance(result, torch.Tensor):
            x.append(result)
    return x


def get_correct_input_sizes(input_size: INPUT_SIZE_TYPE) -> CORRECTED_INPUT_SIZE_TYPE:
    """
    Convert input_size to the correct form, which is a list of tuples.
    Also handles multiple inputs to the network.
    """

    def flatten(nested_array: INPUT_SIZE_TYPE) -> Generator:
        """ Flattens a nested array. """
        for item in nested_array:
            if isinstance(item, (list, tuple)):
                yield from flatten(item)
            else:
                yield item

    assert input_size
    assert all(size > 0 for size in flatten(input_size)), "Negative size found in input_data."

    if isinstance(input_size, list) and isinstance(input_size[0], int):
        return [tuple(input_size)]
    if isinstance(input_size, list):
        return input_size
    if isinstance(input_size, tuple) and isinstance(input_size[0], tuple):
        return list(input_size)
    return [input_size]


def apply_hooks(
    module: nn.Module,
    orig_model: nn.Module,
    batch_dim: int,
    depth: int,
    summary_list: List[LayerInfo],
    idx: Dict[int, int],
    hooks: Optional[List[RemovableHandle]],
    curr_depth: int = 0,
    parent_info: Optional[LayerInfo] = None,
) -> None:
    """
    If input_data is provided, recursively adds hooks to all layers of the model.
    Else, fills summary_list with layer info without computing a forward pass through the network.
    """
    info = LayerInfo(module, curr_depth, None, parent_info)

    def pre_hook(module: nn.Module, inputs: Any) -> None:
        """ Create a LayerInfo object to aggregate information about that layer. """
        del inputs
        nonlocal info
        idx[curr_depth] = idx.get(curr_depth, 0) + 1
        info = LayerInfo(module, curr_depth, idx[curr_depth], parent_info)
        info.depth_index = idx[curr_depth]
        info.check_recursive(summary_list)
        summary_list.append(info)

    def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
        """ Update LayerInfo after forward pass. """
        del module
        info.input_size = info.calculate_size(inputs, batch_dim)
        info.output_size = info.calculate_size(outputs, batch_dim)
        info.calculate_num_params()
        info.executed = True

    submodules = [m for m in module.modules() if m is not orig_model]
    if module != orig_model or isinstance(module, LAYER_MODULES) or not submodules:
        if hooks is None:
            pre_hook(module, None)
        else:
            hooks.append(module.register_forward_pre_hook(pre_hook))
            hooks.append(module.register_forward_hook(hook))

    if curr_depth <= depth:
        for child in module.children():
            apply_hooks(
                child, orig_model, batch_dim, depth, summary_list, idx, hooks, curr_depth + 1, info
            )
