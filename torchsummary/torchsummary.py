""" torchsummary.py """
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import torch


# Some modules do the computation themselves using parameters
# or the parameters of children. Treat these as layers.
LAYER_MODULES = (torch.nn.MultiheadAttention,)

#'kernel_size',
def summary(model, input_size, *args, use_branching=True, max_depth=3, verbose=False,
            col_names=['output_size', 'num_params'], col_width=25, dtypes=None, **kwargs):
    """
    Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)
    Args:
        model (Module): Model to summarize
        x (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function
    """
    formatting = FormattingOptions(use_branching, max_depth, verbose, col_names, col_width)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtypes is None:
        dtypes = [torch.FloatTensor] * len(input_size)

    input_size = get_correct_input_sizes(input_size)

    summary_list = []
    hooks = []
    idx = {}

    def register_hook(module, depth):
        """ Register a hook to get layer info. """
        def hook(module, inputs, outputs):
            """ Create a LayerInfo object to aggregate information about that layer. """
            idx[depth] = idx.get(depth, 0) + 1
            info = LayerInfo(module, depth, idx[depth])
            info.calculate_output_size(outputs)
            info.calculate_num_params()
            info.check_recursive(summary_list)
            summary_list.append(info)

        # ignore Sequential and ModuleList and other containers
        submodules = [m for m in module.modules() if m is not model]
        if isinstance(module, LAYER_MODULES) or module != model or \
            (module == model and not submodules):
            hooks.append(module.register_forward_hook(hook))

    apply_hooks(model, register_hook)

    # batch_size of 2 for batchnorm
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

    return print_results(summary_list, input_size, formatting)


class FormattingOptions:
    """ Class that holds information about formatting the table output. """
    def __init__(self, use_branching, max_depth, verbose, col_names, col_width):
        self.use_branching = use_branching
        self.max_depth = max_depth
        self.verbose = verbose
        self.col_names = col_names
        self.col_width = col_width


class LayerInfo:
    """ Class that holds information about a layer module. """
    def __init__(self, module, depth, depth_index):
        # Identifying information
        self.layer_id = id(module)
        self.module = module
        self.class_name = str(module.__class__).split(".")[-1].split("'")[0]
        self.inner_layers = OrderedDict()
        self.depth = depth
        self.key_name = f"{self.class_name}: {depth}-{depth_index}"

        # Statistics
        self.trainable = False
        self.is_recursive = False
        self.output_size = None
        self.kernel_size = "--"
        self.num_params = 0
        self.macs = 0

    def calculate_output_size(self, outputs):
        """ Set output_size using the model's outputs. """
        if isinstance(outputs, (list, tuple)):
            try:
                self.output_size = list(outputs[0].size())
            except AttributeError:
                # pack_padded_seq and pad_packed_seq store feature into data attribute
                self.output_size = [[-1] + list(o.data.size())[1:] for o in outputs]
                # self.output_size = list(outputs[0].data.size())
        elif isinstance(outputs, dict):
            self.output_size = [[-1] + list(output.size())[1:] for _, output in outputs]
        else:
            self.output_size = list(outputs.size())
            self.output_size[0] = -1

    def calculate_num_params(self):
        """ Set num_params using the module's parameters.  """
        for name, param in self.module.named_parameters():
            if not any(self.module.children()):
                self.num_params += param.nelement() * param.requires_grad
            self.trainable = param.requires_grad # TODO

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

    def macs_to_str(self):
        """ Convert MACs to string. """
        if self.num_params == 0:
            return "--"
        return f'{self.macs:,}'

    def num_params_to_str(self):
        """ Convert num_params to string. """
        assert self.num_params >= 0
        if self.is_recursive:
            return "(recursive)"
        if self.num_params == 0:
            return "--"
        return f'{self.num_params:,}'

    def layer_info_to_row(self, formatting):
        """ Convert layer_info to string representation of a row. """

        def get_start_str(depth):
            return "├─" if depth == 1 else "|    " * (depth - 1) + "└─"

        mapping = {
            'kernel_size': str(self.kernel_size),
            'output_size': str(self.output_size),
            'num_params': self.num_params_to_str(),
            'mult_adds': self.macs_to_str()
        }
        info_to_format = []
        for row_type in formatting.col_names:
            info_to_format.append(mapping[row_type])

        name = self.key_name
        if formatting.use_branching:
            name = get_start_str(self.depth) + name

        new_line = format_row(name, info_to_format, formatting)
        if formatting.verbose:
            for inner_name, inner_shape in self.inner_layers.items():
                if formatting.use_branching:
                    newline += f'{get_start_str(self.depth + 1)}'
                new_line += f"  {inner_name:<13} {str(inner_shape):>20}\n"
        return new_line


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
    # multiple inputs to the network, make sure everything passed in is a list of tuple sizes.
    if isinstance(input_size, tuple):
        return [input_size]
    if isinstance(input_size, list) and isinstance(input_size[0], int):
        return [tuple(input_size)]
    if isinstance(input_size, tuple) and isinstance(input_size[0], tuple):
        return list(input_size)
    return input_size


def apply_hooks(model, register_hook_fn, depth=0):
    """ Recursively adds hooks to all layers of the model. """
    register_hook_fn(model, depth)
    for module in model.children():
        apply_hooks(module, register_hook_fn, depth + 1)


def format_row(layer_name, info_to_use, formatting):
    """ Get the string representation of a single layer of the model. """
    new_line = f'{layer_name:<40} '
    for info in info_to_use:
        new_line += f'{info:<{formatting.col_width}} '
    return new_line.rstrip() + '\n'


def print_layer_tree(summary_list, formatting):
    """ Print each layer of the model using a fancy branching diagram. """
    def _print_layer_tree(summary_list, left=0, right=len(summary_list), depth=1):
        if depth > formatting.max_depth:
            return ''

        new_left = left - 1
        new_str = ''
        for i in range(left, right):
            layer_info = summary_list[i]
            if layer_info.depth == depth:
                new_str += layer_info.layer_info_to_row(formatting) \
                           + _print_layer_tree(summary_list, new_left + 1, i, depth + 1)
                new_left = i
        return new_str

    return _print_layer_tree(summary_list)


def print_layer_list(summary_list, formatting):
    layer_rows = ""
    for layer_info in summary_list:
        layer_rows += layer_info.layer_info_to_row(formatting)
    return layer_rows


def print_results(summary_list, input_size, formatting):
    """ Print results of the summary. """
    def to_megabytes(num):
        """ Converts a float (4 bytes) to megabytes. """
        return abs(num * 4. / (1024 ** 2.))

    total_params, total_output, trainable_params, total_mult_adds = 0, 0, 0, 0
    for layer_info in summary_list:
        if not layer_info.is_recursive:
            total_params += layer_info.num_params
            if layer_info.trainable:
                trainable_params += layer_info.num_params
            if layer_info.num_params > 0:
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
        "mult_adds": total_mult_adds
    }

    header_mapping = {
        'kernel_size': 'Kernel Shape',
        'output_size': 'Output Shape',
        'num_params': 'Param #',
        'mult_adds': 'Mult-Adds'
    }
    headers = [header_mapping[x] for x in formatting.col_names]
    header_row = format_row('Layer (type:depth-idx)', headers, formatting)
    if formatting.use_branching:
        layer_rows = print_layer_tree(summary_list, formatting)
    else:
        layer_rows = print_layer_list(summary_list, formatting)

    width = len(formatting.col_names) * formatting.col_width + 40
    summary_str = (
        f"{'-' * width}\n"
        f"{header_row}"
        f"{'=' * width}\n"
        f"{layer_rows}"
        f"{'=' * width}\n"
        f"Total params: {total_params:,}\n"
        f"Trainable params: {trainable_params:,}\n"
        f"Non-trainable params: {total_params - trainable_params:,}\n"
        f"{'-' * width}\n"
        f"Input size (MB): {total_input_size:0.2f}\n"
        f"Forward/backward pass size (MB): {total_output_size:0.2f}\n"
        f"Params size (MB): {total_params_size:0.2f}\n"
        f"Estimated Total Size (MB): {total_size:0.2f}\n"
        f"{'-' * width}\n"
    )
    print(summary_str)
    return summary_list, SimpleNamespace(**results)
