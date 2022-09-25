from __future__ import annotations

import sys
import warnings
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)

import torch
from torch import nn
from torch.jit import ScriptModule
from torch.utils.hooks import RemovableHandle

from .enums import ColumnSettings, Mode, RowSettings, Verbosity
from .formatting import FormattingOptions
from .layer_info import LayerInfo, get_children_layers, prod
from .model_statistics import ModelStatistics

# Some modules do the computation themselves using parameters
# or the parameters of children. Treat these as layers.
# TODO: figure out a test case for this
LAYER_MODULES = (torch.nn.MultiheadAttention,)
# These modules are not recorded during a forward pass. Handle them separately.
WRAPPER_MODULES = (ScriptModule,)

INPUT_DATA_TYPE = Union[torch.Tensor, Sequence[Any], Mapping[str, Any]]
CORRECTED_INPUT_DATA_TYPE = Optional[Union[Iterable[Any], Mapping[Any, Any]]]
INPUT_SIZE_TYPE = Sequence[Union[int, Sequence[Any], torch.Size]]
CORRECTED_INPUT_SIZE_TYPE = List[Union[Sequence[Any], torch.Size]]

DEFAULT_COLUMN_NAMES = (ColumnSettings.OUTPUT_SIZE, ColumnSettings.NUM_PARAMS)
DEFAULT_ROW_SETTINGS = {RowSettings.DEPTH}
REQUIRES_INPUT = {
    ColumnSettings.INPUT_SIZE,
    ColumnSettings.OUTPUT_SIZE,
    ColumnSettings.MULT_ADDS,
}

_cached_forward_pass: dict[str, list[LayerInfo]] = {}


def summary(
    model: nn.Module,
    input_size: INPUT_SIZE_TYPE | None = None,
    input_data: INPUT_DATA_TYPE | None = None,
    batch_dim: int | None = None,
    cache_forward_pass: bool | None = None,
    col_names: Iterable[str] | None = None,
    col_width: int = 25,
    depth: int = 3,
    device: torch.device | str | None = None,
    dtypes: list[torch.dtype] | None = None,
    mode: str | None = None,
    row_settings: Iterable[str] | None = None,
    verbose: int | None = None,
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
    input_data_specified = input_data is not None or input_size is not None
    if col_names is None:
        columns = (
            DEFAULT_COLUMN_NAMES
            if input_data_specified
            else (ColumnSettings.NUM_PARAMS,)
        )
    else:
        columns = tuple(ColumnSettings(name) for name in col_names)

    if row_settings is None:
        rows = DEFAULT_ROW_SETTINGS
    else:
        rows = {RowSettings(name) for name in row_settings}

    if mode is None:
        model_mode = Mode.EVAL
    else:
        model_mode = Mode(mode)

    if verbose is None:
        # pylint: disable=no-member
        verbose = 0 if hasattr(sys, "ps1") and sys.ps1 else 1

    if cache_forward_pass is None:
        # In the future, this may be enabled by default in Jupyter Notebooks
        cache_forward_pass = False

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validate_user_params(
        input_data, input_size, columns, col_width, device, dtypes, verbose
    )

    x, correct_input_size = process_input(
        input_data, input_size, batch_dim, device, dtypes
    )
    summary_list = forward_pass(
        model, x, batch_dim, cache_forward_pass, device, model_mode, **kwargs
    )
    formatting = FormattingOptions(depth, verbose, columns, col_width, rows)
    results = ModelStatistics(
        summary_list, correct_input_size, get_total_memory_used(x), formatting
    )
    if verbose > Verbosity.QUIET:
        print(results)
    return results


def process_input(
    input_data: INPUT_DATA_TYPE | None,
    input_size: INPUT_SIZE_TYPE | None,
    batch_dim: int | None,
    device: torch.device | str,
    dtypes: list[torch.dtype] | None = None,
) -> tuple[CORRECTED_INPUT_DATA_TYPE, Any]:
    """Reads sample input data to get the input size."""
    x = None
    correct_input_size = []
    if input_data is not None:
        correct_input_size = get_input_data_sizes(input_data)
        x = set_device(input_data, device)
        if isinstance(x, torch.Tensor):
            x = [x]

    if input_size is not None:
        if dtypes is None:
            dtypes = [torch.float] * len(input_size)
        correct_input_size = get_correct_input_sizes(input_size)
        x = get_input_tensor(correct_input_size, batch_dim, dtypes, device)
    return x, correct_input_size


def forward_pass(
    model: nn.Module,
    x: CORRECTED_INPUT_DATA_TYPE,
    batch_dim: int | None,
    cache_forward_pass: bool,
    device: torch.device | str,
    mode: Mode,
    **kwargs: Any,
) -> list[LayerInfo]:
    """Perform a forward pass on the model using forward hooks."""
    global _cached_forward_pass  # pylint: disable=global-variable-not-assigned
    model_name = model.__class__.__name__
    if cache_forward_pass and model_name in _cached_forward_pass:
        return _cached_forward_pass[model_name]

    summary_list, _, hooks = apply_hooks(model_name, model, x, batch_dim)
    if x is None:
        set_children_layers(summary_list)
        return summary_list

    kwargs = set_device(kwargs, device)
    saved_model_mode = model.training
    try:
        if mode == Mode.TRAIN:
            model.train()
        elif mode == Mode.EVAL:
            model.eval()
        else:
            raise RuntimeError(
                f"Specified model mode ({list(Mode)}) not recognized: {mode}"
            )

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
        if hooks:
            for pre_hook, hook in hooks.values():
                pre_hook.remove()
                hook.remove()
        model.train(saved_model_mode)

    add_missing_container_layers(summary_list)
    set_children_layers(summary_list)

    _cached_forward_pass[model_name] = summary_list
    return summary_list


def set_children_layers(summary_list: list[LayerInfo]) -> None:
    """Populates the children and depth_index fields of all LayerInfo."""
    idx: dict[int, int] = {}
    for i, layer in enumerate(summary_list):
        idx[layer.depth] = idx.get(layer.depth, 0) + 1
        layer.depth_index = idx[layer.depth]
        layer.children = get_children_layers(summary_list, i)


def add_missing_container_layers(summary_list: list[LayerInfo]) -> None:
    """Finds container modules not in the currently listed hierarchy."""
    layer_ids = {layer.layer_id for layer in summary_list}
    current_hierarchy: dict[int, LayerInfo] = {}
    for idx, layer_info in enumerate(summary_list):
        # to keep track index of current layer
        # after inserting new layers
        rel_idx = 0

        # create full hierarchy of current layer
        hierarchy = {}
        parent = layer_info.parent_info
        while parent is not None and parent.depth > 0:
            hierarchy[parent.depth] = parent
            parent = parent.parent_info

        # show hierarchy if it is not there already
        for d in range(1, layer_info.depth):
            if (
                d not in current_hierarchy
                or current_hierarchy[d].module is not hierarchy[d].module
            ) and hierarchy[d] is not summary_list[idx + rel_idx - 1]:

                hierarchy[d].calculate_num_params()
                hierarchy[d].check_recursive(layer_ids)
                summary_list.insert(idx + rel_idx, hierarchy[d])
                layer_ids.add(hierarchy[d].layer_id)

                current_hierarchy[d] = hierarchy[d]
                rel_idx += 1

        current_hierarchy[layer_info.depth] = layer_info

        # remove deeper hierarchy
        d = layer_info.depth + 1
        while d in current_hierarchy:
            current_hierarchy.pop(d)
            d += 1


def validate_user_params(
    input_data: INPUT_DATA_TYPE | None,
    input_size: INPUT_SIZE_TYPE | None,
    col_names: tuple[ColumnSettings, ...],
    col_width: int,
    device: torch.device | str | None,
    dtypes: list[torch.dtype] | None,
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
    not_allowed = set(col_names) & REQUIRES_INPUT
    if neither_input_specified and not_allowed:
        raise ValueError(
            "You must pass input_data or input_size in order "
            f"to use columns: {not_allowed}"
        )

    if dtypes is not None and any(
        dtype in (torch.float16, torch.bfloat16) for dtype in dtypes
    ):
        if input_size is not None:
            warnings.warn(
                "Half precision is not supported with input_size parameter, and may "
                "output incorrect results. Try passing input_data directly."
            )

        device_str = device.type if isinstance(device, torch.device) else device
        if device_str == "cpu":
            warnings.warn(
                "Half precision is not supported on cpu. Set the `device` field or "
                "pass `input_data` using the correct device."
            )


def traverse_input_data(
    data: Any, action_fn: Callable[..., Any], aggregate_fn: Callable[..., Any]
) -> Any:
    """
    Traverses any type of nested input data. On a tensor, returns the action given by
    action_fn, and afterwards aggregates the results using aggregate_fn.
    """
    if isinstance(data, torch.Tensor):
        return action_fn(data)

    # Recursively apply to collection items
    aggregate = aggregate_fn(data)
    if isinstance(data, Mapping):
        return aggregate(
            {
                k: traverse_input_data(v, action_fn, aggregate_fn)
                for k, v in data.items()
            }
        )
    if isinstance(data, tuple) and hasattr(data, "_fields"):  # Named tuple
        return aggregate(
            *(traverse_input_data(d, action_fn, aggregate_fn) for d in data)
        )
    if isinstance(data, Iterable) and not isinstance(data, str):
        return aggregate(
            [traverse_input_data(d, action_fn, aggregate_fn) for d in data]
        )
    # Data is neither a tensor nor a collection
    return data


def set_device(data: Any, device: torch.device | str) -> Any:
    """Sets device for all input types and collections of input types."""
    return traverse_input_data(
        data,
        action_fn=lambda data: data.to(device, non_blocking=True),
        aggregate_fn=type,
    )


def get_input_data_sizes(data: Any) -> Any:
    """
    Converts input data to an equivalent data structure of torch.Sizes
    instead of tensors.
    """
    return traverse_input_data(
        data, action_fn=lambda data: data.size(), aggregate_fn=type
    )


def get_total_memory_used(data: CORRECTED_INPUT_DATA_TYPE) -> int:
    """Calculates the total memory of all tensors stored in data."""
    result = traverse_input_data(
        data,
        action_fn=lambda data: sys.getsizeof(data.storage()),
        aggregate_fn=(
            # We don't need the dictionary keys in this case
            lambda data: (lambda d: sum(d.values()))
            if isinstance(data, Mapping)
            else sum
        ),
    )
    return cast(int, result)


def get_input_tensor(
    input_size: CORRECTED_INPUT_SIZE_TYPE,
    batch_dim: int | None,
    dtypes: list[torch.dtype],
    device: torch.device | str,
) -> list[torch.Tensor]:
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


def construct_pre_hook(
    global_layer_info: dict[int, LayerInfo],
    summary_list: list[LayerInfo],
    layer_ids: set[int],
    var_name: str,
    curr_depth: int,
    parent_info: LayerInfo | None,
) -> Callable[[nn.Module, Any], None]:
    def pre_hook(module: nn.Module, inputs: Any) -> None:
        """Create a LayerInfo object to aggregate layer information."""
        del inputs
        info = LayerInfo(var_name, module, curr_depth, parent_info)
        info.calculate_num_params()
        info.check_recursive(layer_ids)
        summary_list.append(info)
        layer_ids.add(info.layer_id)
        global_layer_info[info.layer_id] = info

    return pre_hook


def construct_hook(
    global_layer_info: dict[int, LayerInfo], batch_dim: int | None
) -> Callable[[nn.Module, Any, Any], None]:
    def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
        """Update LayerInfo after forward pass."""
        info = global_layer_info[id(module)]
        if info.contains_lazy_param:
            info.calculate_num_params()
        info.input_size, _ = info.calculate_size(inputs, batch_dim)
        info.output_size, elem_bytes = info.calculate_size(outputs, batch_dim)
        info.output_bytes = elem_bytes * prod(info.output_size)
        info.executed = True
        info.calculate_macs()

    return hook


def apply_hooks(
    model_name: str,
    module: nn.Module,
    input_data: CORRECTED_INPUT_DATA_TYPE,
    batch_dim: int | None,
) -> tuple[
    list[LayerInfo],
    dict[int, LayerInfo],
    dict[int, tuple[RemovableHandle, RemovableHandle]],
]:
    """
    If input_data is provided, recursively adds hooks to all layers of the model.
    Else, fills summary_list with layer info without computing a
    forward pass through the network.
    """
    summary_list: list[LayerInfo] = []
    layer_ids: set[int] = set()  # Used to optimize is_recursive()
    global_layer_info: dict[int, LayerInfo] = {}
    hooks: dict[int, tuple[RemovableHandle, RemovableHandle]] = {}
    stack: list[tuple[str, nn.Module, int, LayerInfo | None]] = [
        (model_name, module, 0, None)
    ]
    while stack:
        var_name, module, curr_depth, parent_info = stack.pop()
        module_id = id(module)

        # Fallback is used if the layer's pre-hook is never called, for example in
        # ModuleLists or Sequentials.
        global_layer_info[module_id] = LayerInfo(
            var_name, module, curr_depth, parent_info
        )
        pre_hook = construct_pre_hook(
            global_layer_info,
            summary_list,
            layer_ids,
            var_name,
            curr_depth,
            parent_info,
        )
        if input_data is None or isinstance(module, WRAPPER_MODULES):
            pre_hook(module, None)
        else:
            # Register the hook using the last layer that uses this module.
            if module_id in hooks:
                for hook in hooks[module_id]:
                    hook.remove()
            hooks[module_id] = (
                module.register_forward_pre_hook(pre_hook),
                module.register_forward_hook(
                    construct_hook(global_layer_info, batch_dim)
                ),
            )

        # Replaces the equivalent recursive call by appending all of the
        # subsequent the module children stack calls in the encountered order.
        # Note: module.named_modules(remove_duplicate=False) doesn't work for
        # some unknown reason (infinite recursion)
        stack += [
            (name, mod, curr_depth + 1, global_layer_info[module_id])
            # pylint: disable=protected-access
            for name, mod in reversed(module._modules.items())
            if mod is not None
        ]
    return summary_list, global_layer_info, hooks


def clear_cached_forward_pass() -> None:
    """Clear the forward pass cache."""
    global _cached_forward_pass  # pylint: disable=global-statement
    _cached_forward_pass = {}
