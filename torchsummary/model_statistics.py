import numpy as np

from .formatting import Verbosity

HEADER_TITLES = {
    "kernel_size": "Kernel Shape",
    "output_size": "Output Shape",
    "num_params": "Param #",
    "mult_adds": "Mult-Adds",
}


class ModelStatistics:
    """ Class for storing results of the summary. """

    def __init__(self, summary_list, input_size, formatting):
        self.summary_list = summary_list
        self.input_size = input_size
        self.total_input = sum([abs(np.prod(sz)) for sz in input_size])
        self.formatting = formatting
        self.total_params, self.trainable_params = 0, 0
        self.total_output, self.total_mult_adds = 0, 0
        for layer_info in summary_list:
            self.total_mult_adds += layer_info.macs
            if not layer_info.is_recursive:
                if layer_info.depth == formatting.max_depth or (
                    not any(layer_info.module.children())
                    and layer_info.depth < formatting.max_depth
                ):
                    self.total_params += layer_info.num_params
                    if layer_info.trainable:
                        self.trainable_params += layer_info.num_params
                if layer_info.num_params > 0 and not any(layer_info.module.children()):
                    # x2 for gradients
                    self.total_output += 2.0 * abs(np.prod(layer_info.output_size))

    @staticmethod
    def to_megabytes(num):
        """ Converts a number (assume floats, 4 bytes each) to megabytes. """
        assert num >= 0
        return num * 4.0 / (1024 ** 2.0)

    def __repr__(self):
        """ Print results of the summary. """
        header_row = self.formatting.format_row("Layer (type:depth-idx)", HEADER_TITLES)
        layer_rows = self.layers_to_str()

        total_size = self.total_input + self.total_output + self.total_params
        width = self.formatting.get_total_width()
        summary_str = (
            "{}\n"
            "{}"
            "{}\n"
            "{}"
            "{}\n"
            "Total params: {:,}\n"
            "Trainable params: {:,}\n"
            "Non-trainable params: {:,}\n"
            # f"Total mult-adds: {self.total_mult_adds:,}\n"
            "{}\n"
            "Input size (MB): {:0.2f}\n"
            "Forward/backward pass size (MB): {:0.2f}\n"
            "Params size (MB): {:0.2f}\n"
            "Estimated Total Size (MB): {:0.2f}\n"
            "{}\n".format(
                ("-" * width),
                (header_row),
                ("=" * width),
                (layer_rows),
                ("=" * width),
                (self.total_params),
                (self.trainable_params),
                (self.total_params - self.trainable_params),
                ("-" * width),
                (self.to_megabytes(self.total_input)),
                (self.to_megabytes(self.total_output)),
                (self.to_megabytes(self.total_params)),
                (self.to_megabytes(total_size)),
                ("-" * width),
            )
        )
        return summary_str

    def layer_info_to_row(self, layer_info, reached_max_depth=False):
        """ Convert layer_info to string representation of a row. """

        def get_start_str(depth):
            return "├─" if depth == 1 else "|    " * (depth - 1) + "└─"

        row_values = {
            "kernel_size": str(layer_info.kernel_size) if layer_info.kernel_size else "--",
            "output_size": str(layer_info.output_size),
            "num_params": layer_info.num_params_to_str(reached_max_depth),
            "mult_adds": layer_info.macs_to_str(reached_max_depth),
        }
        depth = layer_info.depth
        name = (get_start_str(depth) if self.formatting.use_branching else "") + str(layer_info)
        new_line = self.formatting.format_row(name, row_values)
        if self.formatting.verbose == Verbosity.VERBOSE.value:
            for inner_name, inner_shape in layer_info.inner_layers.items():
                prefix = get_start_str(depth + 1) if self.formatting.use_branching else "  "
                extra_row_values = {"kernel_size": str(inner_shape)}
                new_line += self.formatting.format_row(prefix + inner_name, extra_row_values)
        return new_line

    def layers_to_str(self):
        """ Print each layer of the model as tree or as a list. """
        if self.formatting.use_branching:
            return self._layer_tree_to_str()

        layer_rows = ""
        for layer_info in self.summary_list:
            layer_rows += self.layer_info_to_row(layer_info)
        return layer_rows

    def _layer_tree_to_str(self, left=0, right=None, depth=1):
        """ Print each layer of the model using a fancy branching diagram. """
        if depth > self.formatting.max_depth:
            return ""
        new_left = left - 1
        new_str = ""
        if right is None:
            right = len(self.summary_list)
        for i in range(left, right):
            layer_info = self.summary_list[i]
            if layer_info.depth == depth:
                reached_max_depth = depth == self.formatting.max_depth
                new_str += self.layer_info_to_row(layer_info, reached_max_depth)
                new_str += self._layer_tree_to_str(new_left + 1, i, depth + 1)
                new_left = i
        return new_str
