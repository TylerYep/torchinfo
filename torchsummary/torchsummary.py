""" torchsummary.py """
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from torchsummary.formatting import FormattingOptions
from torchsummary.layer_info import LayerInfo
from torchsummary.model_statistics import ModelStatistics

# Some modules do the computation themselves using parameters
# or the parameters of children. Treat these as layers.
LAYER_MODULES = (torch.nn.MultiheadAttention,)  # type: ignore


def summary(
    model: nn.Module,
    input_data: Union[Sequence[torch.Tensor], Sequence[Union[int, Sequence, torch.Size]]],
    *args: Any,
    use_branching: bool = True,
    max_depth: int = 3,
    verbose: int = 1,
    col_names: List[str] = ["output_size", "num_params"],
    col_width: int = 25,
    dtypes: Optional[List[Type[torch.Tensor]]] = None,
    batch_dim: int = 0,
    **kwargs: Any,
) -> ModelStatistics:
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
            default to FloatTensors). NOTE: For scalars, use torch.Size([]).
        use_branching (bool): Whether to use the branching layout for the printed output.
        max_depth (int): number of nested layers to traverse (e.g. Sequentials)
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
        args, kwargs: Other arguments used in `model.forward` function
    """
    assert verbose in (0, 1, 2)
    summary_list: List[LayerInfo] = []
    hooks: List[RemovableHandle] = []
    idx: Dict[int, int] = {}
    apply_hooks(model, model, max_depth, summary_list, hooks, idx, batch_dim)

    if isinstance(input_data, torch.Tensor):
        input_size = get_correct_input_sizes(input_data.size())
        x = [input_data]
    else:
        if dtypes is None:
            dtypes = [torch.FloatTensor] * len(input_data)
        input_size = get_correct_input_sizes(input_data)
        x = get_input_tensor(input_size, batch_dim, dtypes)

    try:
        with torch.no_grad():
            _ = model(*x, *args, **kwargs)
    except Exception:
        print(f"Failed to run torchsummary, printing sizes of executed layers: {summary_list}")
        raise
    finally:
        for hook in hooks:
            hook.remove()

    formatting = FormattingOptions(use_branching, max_depth, verbose, col_names, col_width)
    formatting.set_layer_name_width(summary_list)
    results = ModelStatistics(summary_list, input_size, formatting)
    if verbose > 0:
        print(results)
    return results


def get_input_tensor(input_size, batch_dim, dtypes):
    """ Get input_tensor with batch size 2 for use in model.forward() """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = []
    for size, dtype in zip(input_size, dtypes):
        # add batch_size of 2 for BatchNorm
        if size:
            # Case: input_tensor is a list of dimensions
            input_tensor = torch.rand(*size)
            input_tensor = input_tensor.unsqueeze(dim=batch_dim)
            input_tensor = torch.cat([input_tensor] * 2, dim=batch_dim)
        else:
            # Case: input_tensor is a scalar
            input_tensor = torch.ones(batch_dim)
            input_tensor = torch.cat([input_tensor] * 2, dim=0)
        x.append(input_tensor.type(dtype).to(device))
    return x


def get_correct_input_sizes(input_size):
    """ Convert input_size to the correct form, which is a list of tuples.
    Also handles multiple inputs to the network. """

    def flatten(nested_array):
        """ Flattens a nested array. """
        for item in nested_array:
            if isinstance(item, (list, tuple)):
                yield from flatten(item)
            else:
                yield item

    assert input_size
    assert all(size > 0 for size in flatten(input_size))
    # For multiple inputs to the network, make sure everything passed in is a list of tuple sizes.
    # This code is not very robust, so if you are having trouble here, please submit an issue.
    if isinstance(input_size, tuple) and isinstance(input_size[0], tuple):
        return list(input_size)
    if isinstance(input_size, tuple):
        return [input_size]
    if isinstance(input_size, list) and isinstance(input_size[0], int):
        return [tuple(input_size)]
    return input_size


def apply_hooks(module, orig_model, max_depth, summary_list, hooks, idx, batch_dim, depth=0):
    """ Recursively adds hooks to all layers of the model. """

    def hook(module, inputs, outputs):
        """ Create a LayerInfo object to aggregate information about that layer. """
        del inputs
        idx[depth] = idx.get(depth, 0) + 1
        info = LayerInfo(module, depth, idx[depth])
        info.calculate_output_size(outputs, batch_dim)
        info.calculate_num_params()
        info.check_recursive(summary_list)
        summary_list.append(info)

    # ignore Sequential and ModuleList and other containers
    submodules = [m for m in module.modules() if m is not orig_model]
    if module != orig_model or isinstance(module, LAYER_MODULES) or not submodules:
        hooks.append(module.register_forward_hook(hook))

    if depth <= max_depth:
        for child in module.children():
            apply_hooks(
                child, orig_model, max_depth, summary_list, hooks, idx, batch_dim, depth + 1
            )
