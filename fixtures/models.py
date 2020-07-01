""" fixtures/models.py """
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class SingleInputNet(nn.Module):
    """ Simple CNN model. """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MultipleInputNetDifferentDtypes(nn.Module):
    """ Model with multiple inputs containing different dtypes. """

    def __init__(self):
        super().__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)

        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = x2.type(torch.float)
        x2 = F.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = torch.cat((x1, x2), 0)
        return F.log_softmax(x, dim=1)


class ScalarNet(nn.Module):
    """ Model that takes a scalar as a parameter. """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)

    def forward(self, x, scalar):
        if scalar == 5:
            out = self.conv1(x)
        else:
            out = self.conv2(x)
        return out


class LSTMNet(nn.Module):
    """ Batch-first LSTM model. """

    def __init__(self, vocab_size=20, embed_dim=300, hidden_dim=512, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        out, hidden = self.encoder(embed)
        out = self.decoder(out)
        out = out.view(-1, out.size(2))
        return out, hidden


class RecursiveNet(nn.Module):
    """ Model that uses a layer recursively in computation. """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x, args1=None, args2=None):
        del args1, args2
        for _ in range(3):
            out = self.conv1(x)
            out = self.conv1(out)
        return out


class CustomModule(nn.Module):
    """ Model that defines a custom parameter. """

    def __init__(self, input_size, attention_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((attention_size, input_size)), True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        del x
        return self.weight


class SiameseNets(nn.Module):
    """ Model with MaxPool and ReLU layers. """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)

        self.pooling = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x1, x2):
        x1 = self.pooling(F.relu(self.conv1(x1)))
        x1 = self.pooling(F.relu(self.conv2(x1)))
        x1 = self.pooling(F.relu(self.conv3(x1)))
        x1 = self.pooling(F.relu(self.conv4(x1)))

        x2 = self.pooling(F.relu(self.conv1(x2)))
        x2 = self.pooling(F.relu(self.conv2(x2)))
        x2 = self.pooling(F.relu(self.conv3(x2)))
        x2 = self.pooling(F.relu(self.conv4(x2)))

        batch_size = x1.size(0)
        x1 = x1.view(batch_size, -1)
        x2 = x2.view(batch_size, -1)
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)

        metric = torch.abs(x1 - x2)
        similarity = torch.sigmoid(self.fc2(self.dropout(metric)))
        return similarity


class FunctionalNet(nn.Module):
    """ Model that uses many functional torch layers. """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout2d(0.4)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2048)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class ReturnDictLayer(nn.Module):
    """ Model that returns a dict in forward(). """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        activation_dict = {}
        x = self.conv1(x)
        activation_dict["conv1"] = x
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        activation_dict["conv2"] = x
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        activation_dict["fc1"] = x
        x = self.fc2(x)
        activation_dict["fc2"] = x
        x = F.log_softmax(x, dim=1)
        activation_dict["output"] = x
        return activation_dict


class ReturnDict(nn.Module):
    """ Model that uses a ReturnDictLayer. """

    def __init__(self):
        super().__init__()
        self.return_dict = ReturnDictLayer()

    def forward(self, x, y):
        del y
        activation_dict = self.return_dict(x)
        return activation_dict


class LayerWithRidiculouslyLongNameAndDoesntDoAnything(nn.Module):
    """ Model with a very long name. """

    def forward(self, x):
        return x


class EdgeCaseModel(nn.Module):
    """ Model that throws an exception when used. """

    def __init__(self, throw_error=False, return_str=False):
        super().__init__()
        self.throw_error = throw_error
        self.return_str = return_str
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.model = LayerWithRidiculouslyLongNameAndDoesntDoAnything()

    def forward(self, x):
        x = self.conv1(x)
        x = self.model("string output" if self.return_str else x)
        if self.throw_error:
            x = self.conv1(x)
        return x


class PackPaddedLSTM(nn.Module):
    """ LSTM model with pack_padded layers. """

    def __init__(self, vocab_size=60, embedding_size=128, output_size=18, hidden_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers=1)
        self.hidden2out = nn.Linear(self.hidden_size, output_size)
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, batch, lengths):
        hidden1 = torch.ones(1, batch.size(-1), self.hidden_size)
        hidden2 = torch.ones(1, batch.size(-1), self.hidden_size)
        embeds = self.embedding(batch)
        packed_input = pack_padded_sequence(embeds, lengths)
        _, (ht, _) = self.lstm(packed_input, (hidden1, hidden2))  # type: ignore
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = F.log_softmax(output, dim=1)
        return output
