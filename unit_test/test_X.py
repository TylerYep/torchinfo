from torchsummary.torchsummaryX import summary
import numpy as np
import torchvision
import torch
from torch import nn
import torch.nn.functional as F

from fixtures.models import SingleInputNet, LSTMNet, RecursiveNet, NetWithArgs


def test_convnet():
    summary(SingleInputNet(), (1, 28, 28))


def test_lstm():
    summary_dict, _ = summary(LSTMNet(), (100,), dtypes=[torch.long]) # [length, batch_size]
    assert len(summary_dict) == 3, 'Should find 3 layers'


def test_recursive():
    summary_dict, (total_params, trainable_params) = summary(RecursiveNet(), (64, 28, 28))
    second_layer = tuple(summary_dict.items())[1]
    print(second_layer[1]['params'])
    
    assert len(summary_dict) == 2, 'Should find 2 layers'
    assert second_layer[1]['params'] == '(recursive)', 'should not count the second layer again'
    assert total_params == 36928
    # assert df_total['Totals']['Mult-Adds'] == 57802752


def test_model_with_args():
    summary(NetWithArgs(), (64, 28, 28), "args1", args2="args2")


def test_resnet():
    model = torchvision.models.resnet50()
    _, (total_params, trainable_params) = summary(model, (3, 224, 224))
    # According to https://arxiv.org/abs/1605.07146, resnet50 has ~25.6 M trainable params.
    # Let's make sure we count them correctly

    np.testing.assert_approx_equal(25.6e6, total_params, significant=3)


# model = torchvision.models.resnext50_32x4d()
# summary(model, torch.zeros(4, 3, 224, 224))
