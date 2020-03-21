# import torch
import torchvision

from torchsummary.torchsummary import summary
# from fixtures.models import SingleInputNet, MultipleInputNet, MultipleInputNetDifferentDtypes, \
#     LSTMNet, RecursiveNet, NetWithArgs, CustomModule


if __name__ == '__main__':
    model = torchvision.models.resnet18()
    input_shape = (3, 64, 64)

    summary(model, input_shape,
            max_depth=2,
            col_names=['output_size', 'num_params', 'kernel_size', 'mult_adds'])
