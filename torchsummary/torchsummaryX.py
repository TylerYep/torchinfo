from collections import OrderedDict
import numpy as np
import torch


WIDTH = 64


def summary(model, input_size, *args, batch_size=-1, dtypes=None, **kwargs):
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
    # Some modules do the computation themselves using parameters or the parameters of children, treat these as layers
    layer_modules = (torch.nn.MultiheadAttention, )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtypes == None:
        dtypes = [torch.FloatTensor] * len(input_size)

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    module_names = get_names_dict(model)
    hooks = []
    summary = OrderedDict()

    def register_hook(module):
        def hook(module, inputs, outputs):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            # Lookup name in a dict that includes parents
            module_name = str(module_idx)
            for name, item in module_names.items():
                if item == module:
                    module_name = name
                    break
            key = f"{class_name}-{module_idx + 1}"

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                try:
                    info["out"] = list(outputs[0].size())
                except AttributeError:
                    # pack_padded_seq and pad_packed_seq store feature into data attribute
                    info["out"] = list(outputs[0].data.size())
            else:
                info["out"] = list(outputs.size())

            info["inner"] = OrderedDict()
            info["params"], info["macs"] = 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement() * param.requires_grad
                info["trainable"] = param.requires_grad

                if name == "weight":
                    # ignore N, C when calculate Mult-Adds in ConvNd
                    if "Conv" in class_name:
                        info["macs"] += int(param.nelement() * np.prod(info["out"][2:]))
                    else:
                        info["macs"] += param.nelement()

                # RNN modules have inner weights such as weight_ih_l0
                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            # if the current module is already-used, mark as "(recursive)"
            # check if this module has params
            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            summary[key] = info

        # ignore Sequential and ModuleList and other containers
        if isinstance(module, layer_modules) or not module._modules:
            hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)
    print(input_size)
    # x = torch.zeros(input_size).unsqueeze(dim=0)
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    try:
        with torch.no_grad():
            model(*x) if not (kwargs or args) else model(*x, *args, **kwargs)
    except Exception:
        print(f"Failed to run torchsummaryX.summary, printing sizes of executed layers: {summary}")
        raise
    finally:
        for hook in hooks:
            hook.remove()

    summary_str = "-" * WIDTH + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "=" * WIDTH + "\n"
    total_params, total_output, trainable_params = 0, 0, 0

    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        param_count = summary[layer]["params"]
        if isinstance(summary[layer]["params"], int):
            param_count = f'{summary[layer]["params"]:,}'
            total_params += summary[layer]["params"]
        line_new = f'{layer:>20}  {str(summary[layer]["out"]):>25} {param_count:>15}'

        total_output += np.prod(summary[layer]["out"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True and isinstance(summary[layer]["params"], int):
                trainable_params += summary[layer]["params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ())) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "=" * WIDTH + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params - trainable_params) + "\n"
    summary_str += "-" * WIDTH + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "-" * WIDTH + "\n"
    print(summary_str)
    # from pprint import pprint
    # pprint(summary)
    # print(total_params, trainable_params)
    return summary, (total_params, trainable_params)

    # df = pd.DataFrame(summary).T
    # df["Mult-Adds"] = pd.to_numeric(df["macs"], errors="coerce")
    # df["Params"] = pd.to_numeric(df["params"], errors="coerce")
    # df["Non-trainable params"] = pd.to_numeric(df["params_nt"], errors="coerce")
    # df = df.rename(columns=dict(out="Output Shape"))
    # df_sum = df.sum()
    # df.index.name = "Layer"

    # df = df[["Output Shape", "Params", "Mult-Adds"]]
    # max_repr_width = 300 # max([len(row) for row in df.to_string().split("\n")])

    # df_total = pd.DataFrame({
    #     "Total params": (df_sum["Params"] + df_sum["params_nt"]),
    #     "Trainable params": df_sum["Params"],
    #     "Non-trainable params": df_sum["params_nt"],
    #     "Mult-Adds": df_sum["Mult-Adds"]
    #     }, index=['Totals']
    # ).T

    # with pd.option_context(
    #     "display.max_rows", 600,
    #     "display.max_columns", 10,
    #     "display.width", max_repr_width,
    #     "display.precision", 2,
    #     "display.float_format", pd.io.formats.format.EngFormatter(use_eng_prefix=True)
    # ):
    #     print("="*max_repr_width)
    #     print(df.replace(np.nan, "-"))
    #     print("-"*max_repr_width)
    #     print(df_total)
    #     print("="*max_repr_width)

    # return df, df_total


def get_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}

    def _get_names(module, parent_name=""):
        for key, m in module.named_children():
            cls_name = str(m.__class__).split(".")[-1].split("'")[0]
            num_named_children = len(list(m.named_children()))
            if num_named_children > 0:
                name = parent_name + "." + key if parent_name else key
            else:
                name = parent_name + "." + cls_name + "_" + key if parent_name else key
            names[name] = m

            if isinstance(m, torch.nn.Module):
                _get_names(m, parent_name=name)

    _get_names(model)
    return names
