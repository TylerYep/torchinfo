""" torchsummary.py """
from collections import OrderedDict
import numpy as np
import torch


# Some modules do the computation themselves using parameters
# or the parameters of children. Treat these as layers.
LAYER_MODULES = (torch.nn.MultiheadAttention,)


def summary(model, input_size, *args, use_branching=True, max_depth=3, verbose=False,
            dtypes=None, **kwargs):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtypes is None:
        dtypes = [torch.FloatTensor] * len(input_size)

    input_size = get_correct_input_sizes(input_size)

    hooks = []
    summary_dict = OrderedDict()
    idx = {}

    def register_hook(module, depth):
        """ Register a hook to get layer info. """
        def hook(module, inputs, outputs):
            """ Create a LayerInfo object to aggregate information about that layer. """
            idx[depth] = idx.get(depth, 0) + 1

            info = LayerInfo(module, depth)
            info.calculate_output_size(outputs)
            info.calculate_num_params()
            info.check_recursive(summary_dict)

            key = f"{info.class_name}: {depth}-{idx[depth]}"
            summary_dict[key] = info

        # ignore Sequential and ModuleList and other containers
        if isinstance(module, LAYER_MODULES) or module != model or \
            (module == model and not module._modules):
            hooks.append(module.register_forward_hook(hook))

    apply_hooks(model, register_hook)

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *size).type(dtype).to(device) for size, dtype in zip(input_size, dtypes)]
    try:
        with torch.no_grad():
            _ = model(*x) if not (kwargs or args) else model(*x, *args, **kwargs)
    except Exception:
        print(f"Failed to run torchsummary, printing sizes of executed layers: {summary_dict}")
        raise
    finally:
        for hook in hooks:
            hook.remove()

    return print_results(summary_dict, input_size, use_branching, max_depth, verbose)


class LayerInfo:
    """ Class that holds information about a layer module. """
    def __init__(self, module, depth):
        self.layer_id = id(module)
        self.module = module
        self.class_name = str(module.__class__).split(".")[-1].split("'")[0]
        self.output_size = None
        self.kernel_size = "--"
        self.num_params = 0
        self.inner_layers = OrderedDict()
        self.depth = depth
        self.macs = 0
        self.trainable = False
        self.is_recursive = False

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

    def check_recursive(self, summary_dict):
        """ if the current module is already-used, mark as (recursive) """
        if list(self.module.named_parameters()):
            for v in summary_dict.values():
                if self.layer_id == v.layer_id:
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


def format_row(layer_name, output_size, param_count):
    """ Get the string representation of a single layer of the model. """
    new_line = f'{layer_name:<40} {str(output_size):<25} {param_count:<15}'
    return new_line.rstrip() + '\n'


def print_layer_tree(summary_dict, max_depth, verbose):
    """ Print each layer of the model using a fancy branching diagram. """
    def get_start_str(depth):
        return "├─" if depth == 1 else "|    " * (depth - 1) + "└─"

    layer_names = list(summary_dict.keys())
    def _print_layer_tree(summary_dict, left=0, right=len(layer_names), depth=1):
        if depth > max_depth:
            return ''
        new_str = ''
        new_left = left - 1
        for i in range(left, right):
            layer = layer_names[i]
            layer_info = summary_dict[layer]
            if layer_info.depth == depth:
                indented_name = get_start_str(depth) + layer
                param_count = layer_info.num_params_to_str()
                new_line = format_row(indented_name, layer_info.output_size, param_count)
                if verbose:
                    for inner_name, inner_shape in layer_info.inner_layers.items():
                        indent = get_start_str(depth + 1)
                        new_line += f"{indent} {inner_name:<13} {str(inner_shape):>20}\n"
                new_str += new_line + _print_layer_tree(summary_dict, new_left + 1, i, depth + 1)
                new_left = i
        return new_str

    return _print_layer_tree(summary_dict)


def print_results(summary_dict, input_size, use_branching, max_depth, verbose, width=90):
    """ Print results of the summary. """
    def to_megabytes(num):
        """ Converts a float (4 bytes) to megabytes. """
        return abs(num * 4. / (1024 ** 2.))

    total_params, total_output, trainable_params = 0, 0, 0
    for layer_info in summary_dict.values():
        if not layer_info.is_recursive:
            total_params += layer_info.num_params
            if layer_info.trainable:
                trainable_params += layer_info.num_params
            if layer_info.num_params > 0:
                total_output += np.prod(layer_info.output_size)

    # assume 4 bytes/number (float on cuda).
    total_input_size = to_megabytes(np.prod(sum(input_size, ())))
    total_output_size = to_megabytes(2. * total_output)  # x2 for gradients
    total_params_size = to_megabytes(total_params)
    total_size = total_params_size + total_output_size + total_input_size

    if use_branching:
        layer_rows = print_layer_tree(summary_dict, max_depth=max_depth, verbose=verbose)
    else:
        layer_rows = ""
        for layer, layer_info in summary_dict.items():
            param_count = layer_info.num_params_to_str()
            new_line = format_row(layer, layer_info.output_size, param_count)
            if verbose:
                for inner_name, inner_shape in layer_info.inner_layers.items():
                    new_line += f"  {inner_name:<13} {str(inner_shape):>20}\n"
            layer_rows += new_line

    summary_str = (
        f"{'-' * width}\n"
        f"{format_row('Layer (type:depth-idx)', 'Output Shape', 'Param #')}"
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
    return summary_dict, (total_params, trainable_params)
