from torchsummaryX import summary
import numpy as np
import torchvision
import torch
from torch import nn
import torch.nn.functional as F


def test_convnet():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    summary(Net(), torch.zeros((1, 1, 28, 28)))

def test_lstm():
    class Net(nn.Module):
        def __init__(self,
                    vocab_size=20, embed_dim=300,
                    hidden_dim=512, num_layers=2):
            super().__init__()

            self.hidden_dim = hidden_dim
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.encoder = nn.LSTM(embed_dim, hidden_dim,
                                num_layers=num_layers)
            self.decoder = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            embed = self.embedding(x)
            out, hidden = self.encoder(embed)
            out = self.decoder(out)
            out = out.view(-1, out.size(2))
            return out, hidden
    inputs = torch.zeros((100, 1), dtype=torch.long) # [length, batch_size]
    df, df_total = summary(Net(), inputs)
    assert df.shape[0] == 3, 'Should find 3 layers'

def test_recursive():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

        def forward(self, x):
            out = self.conv1(x)
            out = self.conv1(out)
            return out
    df, df_total = summary(Net(), torch.zeros((1, 64, 28, 28)))
    assert len(df) == 2, 'Should find 2 layers'
    assert np.isnan(df.iloc[1]['Params']), 'should not count the second layer again'
    assert df_total['Totals']['Total params'] == 36928.0
    assert df_total['Totals']['Mult-Adds'] == 57802752.0


def test_args():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

        def forward(self, x, args1, args2):
            out = self.conv1(x)
            out = self.conv1(out)
            return out
    summary(Net(), torch.zeros((1, 64, 28, 28)), "args1", args2="args2")


def test_resnet():
    model = torchvision.models.resnet50()
    df, df_total = summary(model, torch.zeros(4, 3, 224, 224))
    # According to https://arxiv.org/abs/1605.07146, resnet50 has ~25.6 M trainable params.
    # Lets make sure we count them correctly
    np.testing.assert_approx_equal(25.6e6, df_total['Totals']['Total params'], significant=3)


# model = torchvision.models.resnext50_32x4d()
# summary(model, torch.zeros(4, 3, 224, 224))
