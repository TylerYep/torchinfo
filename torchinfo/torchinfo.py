from __future__ import annotations

import difflib
import sys
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

from .enums import ColumnSettings, RowSettings, Verbosity
from .formatting import FormattingOptions
from .layer_info import LayerInfo
from .model_statistics import ModelStatistics

# Some modules do the computation themselves using parameters
# or the parameters of children. Treat these as layers.
LAYER_MODULES = (torch.nn.MultiheadAttention,)
# These modules are not recorded during a forward pass. Handle them separately.
WRAPPER_MODULES = (nn.ParameterList, nn.ModuleList, ScriptModule)

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

    if verbose is None:
        # pylint: disable=no-member
        verbose = 0 if hasattr(sys, "ps1") and sys.ps1 else 1

    if cache_forward_pass is None:
        # In the future, this may be enabled by default in Jupyter Notebooks
        cache_forward_pass = False

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validate_user_params(input_data, input_size, columns, col_width, verbose)

    x, correct_input_size = process_input(
        input_data, input_size, batch_dim, device, dtypes
    )
    summary_list = forward_pass(
        model, x, batch_dim, cache_forward_pass, device, **kwargs
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
    **kwargs: Any,
) -> list[LayerInfo]:
    """Perform a forward pass on the model using forward hooks."""
    global _cached_forward_pass  # pylint: disable=global-variable-not-assigned
    model_name = model.__class__.__name__
    if cache_forward_pass and model_name in _cached_forward_pass:
        return _cached_forward_pass[model_name]

    all_layers: list[LayerInfo] = []
    summary_list: list[LayerInfo] = []
    hooks: dict[int, tuple[RemovableHandle, RemovableHandle]] | None = (
        None if x is None else {}
    )
    named_module = (model_name, model)
    apply_hooks(named_module, model, batch_dim, summary_list, hooks, all_layers)

    if x is None:
        if not summary_list or summary_list[0].var_name != model_name:
            summary_list.insert(0, LayerInfo("", model, 0))
        set_depth_index(summary_list)
        return summary_list

    kwargs = set_device(kwargs, device)
    saved_model_mode = model.training
    try:
        model.eval()
        with torch.no_grad():  # type: ignore[no-untyped-call]
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
            for pre_hook, hook in hooks.values():
                pre_hook.remove()
                hook.remove()
        model.train(saved_model_mode)

    if not summary_list or summary_list[0].var_name != model_name:
        summary_list.insert(0, LayerInfo("", model, 0))

    add_missing_layers(summary_list, all_layers)
    set_depth_index(summary_list)

    _cached_forward_pass[model_name] = summary_list
    return summary_list


def set_depth_index(summary_list: list[LayerInfo]) -> None:
    idx: dict[int, int] = {}
    for layer in summary_list:
        idx[layer.depth] = idx.get(layer.depth, 0) + 1
        layer.depth_index = idx[layer.depth]


def add_missing_layers(
    summary_list: list[LayerInfo], all_layers: list[LayerInfo]
) -> None:
    """
    Edits sumary_list in place by adding LayerInfos that were not included
    during the pre-hooks or forward pass, but were traversed in all_layers.
    """
    # In the future, we should use layer_id instead.
    a = [layer.class_name for layer in summary_list]
    b = [layer.class_name for layer in all_layers]

    # Only add missing layers if their layer names do not already exist
    # in the original summary_list. This is not ideal but is stable for now.
    if not set(b) - set(a):
        return

    for tag, _, _, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        # Ignore all other layer types besides "insert".
        if tag == "insert":
            for i, info in enumerate(all_layers[j1:j2]):
                info.calculate_num_params()
                info.check_recursive(summary_list)
                summary_list.insert(j1 + i, info)


def validate_user_params(
    input_data: INPUT_DATA_TYPE | None,
    input_size: INPUT_SIZE_TYPE | None,
    col_names: tuple[ColumnSettings, ...],
    col_width: int,
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


def apply_hooks(
    named_module: tuple[str, nn.Module],
    orig_model: nn.Module,
    batch_dim: int | None,
    summary_list: list[LayerInfo],
    hooks: dict[int, tuple[RemovableHandle, RemovableHandle]] | None,
    all_layers: list[LayerInfo],
    curr_depth: int = 0,
    parent_info: LayerInfo | None = None,
) -> None:
    """
    If input_data is provided, recursively adds hooks to all layers of the model.
    Else, fills summary_list with layer info without computing a
    forward pass through the network.
    """
    # Fallback is used if the layer's pre-hook is never called, for example in
    # ModuleLists or Sequentials.
    var_name, module = named_module
    info = LayerInfo(var_name, module, curr_depth, parent_info)
    all_layers.append(info)

    def pre_hook(module: nn.Module, inputs: Any) -> None:
        """Create a LayerInfo object to aggregate information about that layer."""
        del inputs
        nonlocal info
        info = LayerInfo(var_name, module, curr_depth, parent_info)
        info.calculate_num_params()
        info.check_recursive(summary_list)
        summary_list.append(info)

    def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
        """Update LayerInfo after forward pass."""
        del module
        info.input_size = info.calculate_size(inputs, batch_dim)
        info.output_size = info.calculate_size(outputs, batch_dim)
        info.executed = True
        info.calculate_macs()

    submodules = [m for m in module.modules() if m is not orig_model]
    if module != orig_model or isinstance(module, LAYER_MODULES) or not submodules:
        if hooks is None or isinstance(module, WRAPPER_MODULES):
            pre_hook(module, None)
        else:
            key = id(module)
            if key not in hooks:
                hooks[key] = (
                    module.register_forward_pre_hook(pre_hook),
                    module.register_forward_hook(hook),
                )

    # module.named_modules(remove_duplicate=False) doesn't work (infinite recursion).
    for name, mod in module._modules.items():  # pylint: disable=protected-access
        assert mod is not None
        child = (name, mod)
        apply_hooks(
            child,
            orig_model,
            batch_dim,
            summary_list,
            hooks,
            all_layers,
            curr_depth + 1,
            info,
        )


def clear_cached_forward_pass() -> None:
    """Clear the forward pass cache."""
    global _cached_forward_pass  # pylint: disable=global-statement
    _cached_forward_pass = {}
