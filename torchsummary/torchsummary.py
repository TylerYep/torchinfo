""" torchsummary.py """
from types import SimpleNamespace
from collections import OrderedDict
import math
import numpy as np
import torch

# Some modules do the computation themselves using parameters
# or the parameters of children. Treat these as layers.
LAYER_MODULES = (torch.nn.MultiheadAttention,)  # type: ignore
HEADER_TITLES = {
    'kernel_size': 'Kernel Shape',
    'output_size': 'Output Shape',
    'num_params': 'Param #',
    'mult_adds': 'Mult-Adds'
}


def summary(model, input_size, *args, use_branching=True, max_depth=3, verbose=False,
            col_names=['output_size', 'num_params'], col_width=25, dtypes=None, **kwargs):
    """
    Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)
    Args:
        model (Module): Model to summarize
        input_size (Tuple): Input tensor of the model with [N, C, H, W] shape
            dtype and device have to match to the model
        use_branching (bool): Whether to use the branching layout for the printed output.
        max_depth (int): number of nested layers to traverse (e.g. Sequentials)
        verbose (bool): Whether to show weight and bias layers in full detail
        col_names (List): columns to show in the output. Currently supported:
            ['output_size', 'num_params', 'kernel_size', 'mult_adds']
        col_width (int): width of each column
        dtypes (List or None): for multiple inputs or args, must specify the size of both inputs.
            You must also specify the types of each parameter here.
        args, kwargs: Other arguments used in `model.forward` function
    """
    summary_list, hooks, idx = [], [], {}
    apply_hooks(model, model, max_depth, summary_list, hooks, idx)

    if dtypes is None:
        dtypes = [torch.FloatTensor] * len(input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = get_correct_input_sizes(input_size)

    # batch_size of 2 for BatchNorm
    x = [torch.rand(2, *size).type(dtype).to(device) for size, dtype in zip(input_size, dtypes)]
    try:
        with torch.no_grad():
            _ = model(*x) if not (kwargs or args) else model(*x, *args, **kwargs)
    except Exception:
        print(f"Failed to run torchsummary, printing sizes of executed layers: {summary_list}")
        raise
    finally:
        for hook in hooks:
            hook.remove()

    formatting = FormattingOptions(use_branching, max_depth, verbose, col_names, col_width)
    formatting.set_layer_name_width(summary_list)
    return print_results(summary_list, input_size, formatting)


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


def apply_hooks(module, orig_model, max_depth, summary_list, hooks, idx, depth=0):
    """ Recursively adds hooks to all layers of the model. """

    def hook(module, inputs, outputs):
        """ Create a LayerInfo object to aggregate information about that layer. """
        idx[depth] = idx.get(depth, 0) + 1
        info = LayerInfo(module, depth, idx[depth])
        info.calculate_output_size(outputs)
        info.calculate_num_params()
        info.check_recursive(summary_list)
        summary_list.append(info)

    # ignore Sequential and ModuleList and other containers
    submodules = [m for m in module.modules() if m is not orig_model]
    if module != orig_model or isinstance(module, LAYER_MODULES) or not submodules:
        hooks.append(module.register_forward_hook(hook))

    # if depth <= max_depth:
    for child in module.children():
        apply_hooks(child, orig_model, max_depth, summary_list, hooks, idx, depth + 1)


class FormattingOptions:
    """ Class that holds information about formatting the table output. """
    def __init__(self, use_branching, max_depth, verbose, col_names, col_width):
        self.use_branching = use_branching
        self.max_depth = max_depth
        self.verbose = verbose
        self.col_names = col_names
        self.col_width = col_width
        self.layer_name_width = 40

    def set_layer_name_width(self, summary_list):
        """ Set layer name width by taking the longest line length and rounding up to
        the nearest multiple of 5. """
        max_length = 0
        for info in summary_list:
            str_len = len(str(info))
            depth_indent = info.depth * 5 + 1
            max_length = max(max_length, str_len + depth_indent)
        if max_length >= self.layer_name_width:
            self.layer_name_width = math.ceil(max_length / 5.) * 5

    def get_total_width(self):
        """ Calculate the total width of all lines in the table. """
        return len(self.col_names) * self.col_width + self.layer_name_width


class LayerInfo:
    """ Class that holds information about a layer module. """
    def __init__(self, module, depth, depth_index):
        # Identifying information
        self.layer_id = id(module)
        self.module = module
        self.class_name = str(module.__class__).split(".")[-1].split("'")[0]
        self.inner_layers = OrderedDict()
        self.depth = depth
        self.depth_index = depth_index

        # Statistics
        self.trainable = True
        self.is_recursive = False
        self.output_size = None
        self.kernel_size = "--"
        self.num_params = 0
        self.macs = 0

    def __repr__(self):
        return f"{self.class_name}: {self.depth}-{self.depth_index}"

    def calculate_output_size(self, outputs):
        """ Set output_size using the model's outputs. """
        if isinstance(outputs, (list, tuple)):
            try:
                self.output_size = list(outputs[0].size())
            except AttributeError:
                # pack_padded_seq and pad_packed_seq store feature into data attribute
                # self.output_size = list(outputs[0].data.size())
                self.output_size = [[-1] + list(o.data.size())[1:] for o in outputs]
        elif isinstance(outputs, dict):
            self.output_size = [[-1] + list(output.size())[1:] for _, output in outputs]
        else:
            self.output_size = list(outputs.size())
            self.output_size[0] = -1

    def calculate_num_params(self):
        """ Set num_params using the module's parameters.  """
        for name, param in self.module.named_parameters():
            self.num_params += param.nelement()
            self.trainable &= param.requires_grad

            if name == "weight":
                ksize = list(param.size())
                # to make [in_shape, out_shape, ksize, ksize]
                if len(ksize) > 1:
                    ksize[0], ksize[1] = ksize[1], ksize[0]
                self.kernel_size = ksize

                # ignore N, C when calculate Mult-Adds in ConvNd
                if "Conv" in self.class_name:
                    self.macs += int(param.nelement() * np.prod(self.output_size[2:]))
                else:
                    self.macs += param.nelement()

            # RNN modules have inner weights such as weight_ih_l0
            elif "weight" in name:
                self.inner_layers[name] = list(param.size())
                self.macs += param.nelement()

    def check_recursive(self, summary_list):
        """ if the current module is already-used, mark as (recursive).
        Must check before adding line to the summary. """
        if list(self.module.named_parameters()):
            for other_layer in summary_list:
                if self.layer_id == other_layer.layer_id:
                    self.is_recursive = True

    def macs_to_str(self, reached_max_depth):
        """ Convert MACs to string. """
        if self.num_params > 0 and (reached_max_depth or not any(self.module.children())):
            return f'{self.macs:,}'
        return "--"

    def num_params_to_str(self, reached_max_depth=False):
        """ Convert num_params to string. """
        assert self.num_params >= 0
        if self.is_recursive:
            return "(recursive)"
        if self.num_params > 0:
            param_count_str = f'{self.num_params:,}'
            if reached_max_depth or not any(self.module.children()):
                if not self.trainable:
                    return f'({param_count_str})'
                return param_count_str
        return "--"

    def layer_info_to_row(self, formatting, reached_max_depth=False):
        """ Convert layer_info to string representation of a row. """

        def get_start_str(depth):
            return "├─" if depth == 1 else "|    " * (depth - 1) + "└─"

        row_values = {
            'kernel_size': str(self.kernel_size),
            'output_size': str(self.output_size),
            'num_params': self.num_params_to_str(reached_max_depth),
            'mult_adds': self.macs_to_str(reached_max_depth)
        }
        name = str(self)
        if formatting.use_branching:
            name = get_start_str(self.depth) + name
        new_line = format_row(name, row_values, formatting)
        if formatting.verbose:
            for inner_name, inner_shape in self.inner_layers.items():
                prefix = get_start_str(self.depth + 1) if formatting.use_branching else '  '
                extra_row_values = {'kernel_size': str(inner_shape)}
                new_line += format_row(prefix + inner_name, extra_row_values, formatting)
        return new_line


def format_row(layer_name, row_values, formatting):
    """ Get the string representation of a single layer of the model. """
    info_to_use = [row_values.get(row_type, "") for row_type in formatting.col_names]
    new_line = f'{layer_name:<{formatting.layer_name_width}} '
    for info in info_to_use:
        new_line += f'{info:<{formatting.col_width}} '
    return new_line.rstrip() + '\n'


def layer_tree_to_str(summary_list, formatting, left=0, right=None, depth=1):
    """ Print each layer of the model using a fancy branching diagram. """
    if depth > formatting.max_depth:
        return ''
    new_left = left - 1
    new_str = ''
    if right is None:
        right = len(summary_list)
    for i in range(left, right):
        layer_info = summary_list[i]
        if layer_info.depth == depth:
            reached_max_depth = depth == formatting.max_depth
            new_str += layer_info.layer_info_to_row(formatting, reached_max_depth) \
                + layer_tree_to_str(summary_list, formatting, new_left + 1, i, depth + 1)
            new_left = i
    return new_str


def layer_list_to_str(summary_list, formatting):
    layer_rows = ""
    for layer_info in summary_list:
        layer_rows += layer_info.layer_info_to_row(formatting)
    return layer_rows


def calculate_results(summary_list, input_size, formatting):
    def to_megabytes(num):
        """ Converts a number of floats (4 bytes each) to megabytes. """
        return abs(num * 4. / (1024 ** 2.))

    total_params, total_output, trainable_params, total_mult_adds = 0, 0, 0, 0
    for layer_info in summary_list:
        if not layer_info.is_recursive:
            if (not any(layer_info.module.children()) and layer_info.depth < formatting.max_depth) \
                    or layer_info.depth == formatting.max_depth:
                total_params += layer_info.num_params
                if layer_info.trainable:
                    trainable_params += layer_info.num_params
            if layer_info.num_params > 0 and not any(layer_info.module.children()):
                total_output += np.prod(layer_info.output_size)
        total_mult_adds += layer_info.macs

    # assume 4 bytes/number (float on cuda).
    total_input_size = to_megabytes(np.prod(sum(input_size, ())))
    total_output_size = to_megabytes(2. * total_output)  # x2 for gradients
    total_params_size = to_megabytes(total_params)
    total_size = total_params_size + total_output_size + total_input_size
    results = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "mult_adds": total_mult_adds,
        "total_input_size": total_input_size,
        "total_output_size": total_output_size,
        "total_params_size": total_params_size,
        "total_size": total_size,
        "total_mult_adds": total_mult_adds
    }
    return SimpleNamespace(**results)


def print_results(summary_list, input_size, formatting):
    """ Print results of the summary. """
    results = calculate_results(summary_list, input_size, formatting)
    header_row = format_row('Layer (type:depth-idx)', HEADER_TITLES, formatting)
    if formatting.use_branching:
        layer_rows = layer_tree_to_str(summary_list, formatting)
    else:
        layer_rows = layer_list_to_str(summary_list, formatting)

    width = formatting.get_total_width()
    summary_str = (
        f"{'-' * width}\n"
        f"{header_row}"
        f"{'=' * width}\n"
        f"{layer_rows}"
        f"{'=' * width}\n"
        f"Total params: {results.total_params:,}\n"
        f"Trainable params: {results.trainable_params:,}\n"
        f"Non-trainable params: {results.total_params - results.trainable_params:,}\n"
        # f"Total mult-adds: {results.total_mult_adds:,}\n"
        f"{'-' * width}\n"
        f"Input size (MB): {results.total_input_size:0.2f}\n"
        f"Forward/backward pass size (MB): {results.total_output_size:0.2f}\n"
        f"Params size (MB): {results.total_params_size:0.2f}\n"
        f"Estimated Total Size (MB): {results.total_size:0.2f}\n"
        f"{'-' * width}\n"
    )
    print(summary_str)
    return summary_list, results
