import math


class FormattingOptions:
    """ Class that holds information about formatting the table output. """

    def __init__(self, use_branching, max_depth, verbose, col_names, col_width):
        self.use_branching = use_branching
        self.max_depth = max_depth
        self.verbose = verbose
        self.col_names = col_names
        self.col_width = col_width
        self.layer_name_width = 40

    def set_layer_name_width(self, summary_list, align_val=5):
        """ Set layer name width by taking the longest line length and rounding up to
        the nearest multiple of align_val. """
        max_length = 0
        for info in summary_list:
            depth_indent = info.depth * align_val + 1
            max_length = max(max_length, len(str(info)) + depth_indent)
        if max_length >= self.layer_name_width:
            self.layer_name_width = math.ceil(max_length / align_val) * align_val

    def get_total_width(self):
        """ Calculate the total width of all lines in the table. """
        return len(self.col_names) * self.col_width + self.layer_name_width

    def format_row(self, layer_name, row_values):
        """ Get the string representation of a single layer of the model. """
        info_to_use = [row_values.get(row_type, "") for row_type in self.col_names]
        new_line = f"{layer_name:<{self.layer_name_width}} "
        for info in info_to_use:
            new_line += f"{info:<{self.col_width}} "
        return new_line.rstrip() + "\n"
