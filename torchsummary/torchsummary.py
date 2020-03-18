from collections import OrderedDict
import numpy as np
import torch


WIDTH = 90


def apply_hooks(model, register_hook_fn, depth=0):
    register_hook_fn(model, depth)
    for module in model.children():
        apply_hooks(module, register_hook_fn, depth + 1)


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            yield from flatten(i)
        else:
            yield i


class LayerInfo:
    def __init__(self, module, depth):
        self.id = id(module)
        self.module = module
        self.class_name = str(module.__class__).split(".")[-1].split("'")[0]
        self.output_size = None
        self.kernel_size = "--"
        self.num_params = 0
        self.inner_layers = OrderedDict()
        self.depth = depth
        self.macs = 0
        self.trainable = False

    def calculate_output_size(self, outputs):
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
        for name, param in self.module.named_parameters():
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

        if self.num_params == 0 or any(self.module.children()):
            self.num_params = "--"
            self.macs = "--"


def get_correct_input_sizes(input_size):
    assert input_size
    assert all(size > 0 for size in flatten(input_size))

    # multiple inputs to the network, make sure everything passed in is a list of tuple sizes.
    if isinstance(input_size, tuple):
        return [input_size]
    elif isinstance(input_size, list) and isinstance(input_size[0], int):
        return [tuple(input_size)]
    elif isinstance(input_size, tuple) and isinstance(input_size[0], tuple):
        return list(input_size)

    return input_size



def summary(model, input_size, *args, max_depth=1, dtypes=None, **kwargs):
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
    # Some modules do the computation themselves using parameters or the parameters of children.
    # Treat these as layers.
    LAYER_MODULES = (torch.nn.MultiheadAttention, )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtypes is None:
        dtypes = [torch.FloatTensor] * len(input_size)

    input_size = get_correct_input_sizes(input_size)
    module_names = get_names_dict(model)
    hooks = []
    summary = OrderedDict()
    idx = {}

    def register_hook(module, depth):
        def hook(module, inputs, outputs):
            idx[depth] = idx.get(depth, 0) + 1

            # Lookup name in a dict that includes parents
            # module_name = module_names.get(module, str(len(summary)))

            info = LayerInfo(module, depth)
            info.calculate_output_size(outputs)
            info.calculate_num_params()

            # if the current module is already-used, mark as "(recursive)"
            # check if this module has params
            if list(module.named_parameters()):
                for v in summary.values():
                    if info.id == v.id:
                        info.num_params = "(recursive)"

            key = f"{info.class_name}: {depth}-{idx[depth]}"
            summary[key] = info

        # ignore Sequential and ModuleList and other containers
        if isinstance(module, LAYER_MODULES) or module != model or \
            (module == model and not module._modules):
            hooks.append(module.register_forward_hook(hook))

    apply_hooks(model, register_hook)

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *size).type(dtype).to(device) for size, dtype in zip(input_size, dtypes)]
    try:
        with torch.no_grad():
            model(*x) if not (kwargs or args) else model(*x, *args, **kwargs)
    except Exception:
        print(f"Failed to run torchsummary, printing sizes of executed layers: {summary}")
        raise
    finally:
        for hook in hooks:
            hook.remove()

    return print_results(summary, input_size)


def print_layer_tree(summary, max_depth=3):
    keys = list(summary.keys())
    def _print_layer_tree(summary, left=0, right=len(keys), depth=1):
        if depth > max_depth:
            return ''

        new_str = ''
        nl = left - 1
        for i in range(left, right):
            layer = keys[i]
            layer_info = summary[layer]
            if layer_info.depth == depth:
                if depth == 1:
                    start = "├─" + layer
                else:
                    start = "|    " * (depth - 1) + "└─" + layer

                param_count = layer_info.num_params
                if isinstance(param_count, int):
                    param_count = f'{layer_info.num_params:,}'

                new_line = f'{start:<40} {str(layer_info.output_size):<25} {param_count:<15}'
                new_line = new_line.rstrip() + '\n'
                # for inner_name, inner_shape in layer_info.inner_layers.items():
                #     new_line += f"  {inner_name:<13} {str(inner_shape):>20}\n"
                new_str += new_line + _print_layer_tree(summary, nl+1, i, depth+1)
                nl = i
        return new_str
    return _print_layer_tree(summary)


def print_results(summary, input_size):
    summary_str = "-" * WIDTH + "\n"
    line_new = f'{"Layer (type:depth-idx)":<40} {"Output Shape":<25} {"Param #":<15}'
    summary_str += line_new.rstrip() + "\n"
    summary_str += "=" * WIDTH + "\n"
    summary_str += print_layer_tree(summary)

    total_params, total_output, trainable_params = 0, 0, 0
    for layer, layer_info in summary.items():
        param_count = layer_info.num_params
        if isinstance(param_count, int):
            total_params += param_count
            if layer_info.trainable:
                trainable_params += param_count
            param_count = f'{layer_info.num_params:,}'
            total_output += np.prod(layer_info.output_size)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ())) * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += ("=" * WIDTH + "\n" +
                    f"Total params: {total_params:,}\n"
                    f"Trainable params: {trainable_params:,}\n"
                    f"Non-trainable params: {total_params - trainable_params:,}\n"
                    + "-" * WIDTH + "\n" +
                    f"Input size (MB): {total_input_size:0.2f}\n"
                    f"Forward/backward pass size (MB): {total_output_size:0.2f}\n"
                    f"Params size (MB): {total_params_size:0.2f}\n"
                    f"Estimated Total Size (MB): {total_size:0.2f}\n"
                    + "-" * WIDTH + "\n")
    print(summary_str)
    return summary, (total_params, trainable_params)


def get_names_dict(model):
    """ Recursive walk to get names including path. """
    names = {}
    def _get_names(module, parent_name=""):
        for key, m in module.named_children():
            cls_name = str(m.__class__).split(".")[-1].split("'")[0]
            num_named_children = len(list(m.named_children()))
            if num_named_children > 0:
                name = parent_name + "." + key if parent_name else key
            else:
                name = parent_name + "." + cls_name + "_" + key if parent_name else key
            names[m] = name
            if isinstance(m, torch.nn.Module):
                _get_names(m, parent_name=name)
    _get_names(model)
    return names
