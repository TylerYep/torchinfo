from torchsummary.torchsummaryX import summary
import numpy as np
import torchvision
import torch
from torch import nn
import torch.nn.functional as F

from fixtures.models import SingleInputNet, LSTMNet, RecursiveNet, NetWithArgs


def test_convnet():
    summary(SingleInputNet(), torch.zeros((1, 1, 28, 28)))


def test_lstm():
    inputs = torch.zeros((100, 1), dtype=torch.long) # [length, batch_size]
    df, df_total = summary(LSTMNet(), inputs)
    assert df.shape[0] == 3, 'Should find 3 layers'


def test_recursive():
    df, df_total = summary(RecursiveNet(), torch.zeros((1, 64, 28, 28)))
    assert len(df) == 2, 'Should find 2 layers'
    assert np.isnan(df.iloc[1]['Params']), 'should not count the second layer again'
    assert df_total['Totals']['Total params'] == 36928.0
    assert df_total['Totals']['Mult-Adds'] == 57802752.0


def test_model_with_args():
    summary(NetWithArgs(), torch.zeros((1, 64, 28, 28)), "args1", args2="args2")


def test_resnet():
    model = torchvision.models.resnet50()
    df, df_total = summary(model, torch.zeros(4, 3, 224, 224))
    # According to https://arxiv.org/abs/1605.07146, resnet50 has ~25.6 M trainable params.
    # Let's make sure we count them correctly
    np.testing.assert_approx_equal(25.6e6, df_total['Totals']['Total params'], significant=3)


# model = torchvision.models.resnext50_32x4d()
# summary(model, torch.zeros(4, 3, 224, 224))
