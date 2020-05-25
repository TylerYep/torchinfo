""" torchsummary.py """
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

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


def summary(
    model: nn.Module,
    input_data: Union[torch.Tensor, torch.Size, Sequence[torch.Tensor], INPUT_SIZE_TYPE],
    *args: Any,
    batch_dim: int = 0,
    branching: bool = True,
    col_names: Sequence[str] = ("output_size", "num_params"),
    col_width: int = 25,
    depth: int = 3,
    device: Optional[torch.device] = None,
    dtypes: Optional[List[torch.dtype]] = None,
    verbose: int = 1,
    **kwargs: Any,
) -> ModelStatistics:
    """
    Summarize the given PyTorch model. Summarized information includes:
        1) output shape,
        2) kernel shape,
        3) number of the parameters
        4) operations (Mult-Adds)

    Arguments:
        model (nn.Module): PyTorch model to summarize
        input_data (Sequence of Sizes or Tensors):
            Example input tensor of the model (dtypes inferred from model input).
            - OR -
            Shape of input data as a List/Tuple/torch.Size (dtypes must match model
            input, default is FloatTensors).
        batch_dim (int): batch_dimension of input data
        branching (bool): Whether to use the branching layout for the printed output.
        col_names (Sequence[str]): specify which columns to show in the output.
            Currently supported:
            ('output_size', 'num_params', 'kernel_size', 'mult_adds')
        col_width (int): width of each column
        depth (int): number of nested layers to traverse (e.g. Sequentials)
        device (torch.Device): Uses this torch device for model and input_data.
            Defaults to torch.cuda.is_available().
        dtypes (List[torch.dtype]): for multiple inputs, specify the size of both inputs, and
            also specify the types of each parameter here.
        verbose (int):
            0 (quiet): No output
            1 (default): Print model summary
            2 (verbose): Show weight and bias layers in full detail
        args, kwargs: Other arguments used in `model.forward` function.
    """
    assert verbose in (0, 1, 2)
    summary_list: List[LayerInfo] = []
    hooks: List[RemovableHandle] = []
    idx: Dict[int, int] = {}
    apply_hooks(model, model, depth, summary_list, hooks, idx, batch_dim)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(input_data, torch.Tensor):
        input_size = get_correct_input_sizes(input_data.size())
        x = [input_data.to(device)]

    elif isinstance(input_data, (list, tuple)):
        if all(isinstance(data, torch.Tensor) for data in input_data):
            input_sizes = [data.size() for data in input_data]  # type: ignore
            input_size = get_correct_input_sizes(input_sizes)
            x = [data.to(device) for data in input_data]  # type: ignore
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

    args = tuple([t.to(device) if torch.is_tensor(t) else t for t in args])
    kwargs = {k: kwargs[k].to(device) if torch.is_tensor(kwargs[k]) else k for k in kwargs}

    try:
        with torch.no_grad():
            _ = model.to(device)(*x, *args, **kwargs)
    except Exception:
        print(f"Failed to run torchsummary, printing sizes of executed layers: {summary_list}")
        raise
    finally:
        for hook in hooks:
            hook.remove()

    formatting = FormattingOptions(branching, depth, verbose, col_names, col_width)
    formatting.set_layer_name_width(summary_list)
    results = ModelStatistics(summary_list, input_size, formatting)
    if verbose > Verbosity.QUIET.value:
        print(results)
    return results


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
    """ Convert input_size to the correct form, which is a list of tuples.
    Also handles multiple inputs to the network. """

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
    depth: int,
    summary_list: List[LayerInfo],
    hooks: List[RemovableHandle],
    idx: Dict[int, int],
    batch_dim: int,
    curr_depth: int = 0,
) -> None:
    """ Recursively adds hooks to all layers of the model. """

    def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
        """ Create a LayerInfo object to aggregate information about that layer. """
        del inputs
        idx[curr_depth] = idx.get(curr_depth, 0) + 1
        info = LayerInfo(module, curr_depth, idx[curr_depth])
        info.calculate_output_size(outputs, batch_dim)
        info.calculate_num_params()
        info.check_recursive(summary_list)
        summary_list.append(info)

    # ignore Sequential and ModuleList and other containers
    submodules = [m for m in module.modules() if m is not orig_model]
    if module != orig_model or isinstance(module, LAYER_MODULES) or not submodules:
        hooks.append(module.register_forward_hook(hook))

    if curr_depth <= depth:
        for child in module.children():
            apply_hooks(
                child, orig_model, depth, summary_list, hooks, idx, batch_dim, curr_depth + 1
            )
