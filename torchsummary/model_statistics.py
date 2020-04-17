import numpy as np

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
        layer_rows = self.layers_to_str(self.summary_list, self.formatting)

        total_size = self.total_input + self.total_output + self.total_params
        width = self.formatting.get_total_width()
        summary_str = (
            f"{'-' * width}\n"
            f"{header_row}"
            f"{'=' * width}\n"
            f"{layer_rows}"
            f"{'=' * width}\n"
            f"Total params: {self.total_params:,}\n"
            f"Trainable params: {self.trainable_params:,}\n"
            f"Non-trainable params: {self.total_params - self.trainable_params:,}\n"
            # f"Total mult-adds: {self.total_mult_adds:,}\n"
            f"{'-' * width}\n"
            f"Input size (MB): {self.to_megabytes(self.total_input):0.2f}\n"
            f"Forward/backward pass size (MB): {self.to_megabytes(self.total_output):0.2f}\n"
            f"Params size (MB): {self.to_megabytes(self.total_params):0.2f}\n"
            f"Estimated Total Size (MB): {self.to_megabytes(total_size):0.2f}\n"
            f"{'-' * width}\n"
        )
        return summary_str

    def layers_to_str(self, summary_list, formatting):
        """ Print each layer of the model as tree or as a list. """
        if formatting.use_branching:
            return self._layer_tree_to_str(summary_list, formatting)

        layer_rows = ""
        for layer_info in summary_list:
            layer_rows += layer_info.layer_info_to_row(formatting)
        return layer_rows

    def _layer_tree_to_str(self, summary_list, formatting, left=0, right=None, depth=1):
        """ Print each layer of the model using a fancy branching diagram. """
        if depth > formatting.max_depth:
            return ""
        new_left = left - 1
        new_str = ""
        if right is None:
            right = len(summary_list)
        for i in range(left, right):
            layer_info = summary_list[i]
            if layer_info.depth == depth:
                reached_max_depth = depth == formatting.max_depth
                new_str += layer_info.layer_info_to_row(
                    formatting, reached_max_depth
                ) + self._layer_tree_to_str(summary_list, formatting, new_left + 1, i, depth + 1)
                new_left = i
        return new_str
