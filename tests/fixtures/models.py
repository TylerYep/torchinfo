# pylint: disable=too-few-public-methods
from __future__ import annotations

import math
from collections import namedtuple
from typing import Any, Sequence, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class IdentityModel(nn.Module):
    """Identity Model."""

    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x: Any) -> Any:
        return self.identity(x)


class LinearModel(nn.Module):
    """Linear Model."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x


class UninitializedParameterModel(nn.Module):
    """UninitializedParameter test"""

    def __init__(self) -> None:
        super().__init__()
        self.param: (
            nn.Parameter | nn.UninitializedParameter
        ) = nn.UninitializedParameter()

    def init_param(self) -> None:
        self.param = nn.Parameter(torch.zeros(128))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.init_param()
        return x


class SingleInputNet(nn.Module):
    """Simple CNN model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MultipleInputNetDifferentDtypes(nn.Module):
    """Model with multiple inputs containing different dtypes."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)

        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = x2.type(torch.float)
        x2 = F.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = torch.cat((x1, x2), 0)
        return F.log_softmax(x, dim=1)


class ScalarNet(nn.Module):
    """Model that takes a scalar as a parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)

    def forward(self, x: torch.Tensor, scalar: float) -> torch.Tensor:
        out = x
        if scalar == 5:
            out = self.conv1(out)
        else:
            out = self.conv2(out)
        return out


class LSTMNet(nn.Module):
    """Batch-first LSTM model."""

    def __init__(
        self,
        vocab_size: int = 20,
        embed_dim: int = 300,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # We use batch_first=False here.
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embedding(x)
        out, hidden = self.encoder(embed)
        out = self.decoder(out)
        out = out.view(-1, out.size(2))
        return out, hidden


class RecursiveNet(nn.Module):
    """Model that uses a layer recursively in computation."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(
        self, x: torch.Tensor, args1: Any = None, args2: Any = None
    ) -> torch.Tensor:
        del args1, args2
        out = x
        for _ in range(3):
            out = self.conv1(out)
            out = self.conv1(out)
        return out


class CustomParameter(nn.Module):
    """Model that defines a custom parameter."""

    def __init__(self, input_size: int, attention_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones((attention_size, input_size)), True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        return self.weight


class ParameterListModel(nn.Module):
    """ParameterList of custom parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(weight)
                for weight in torch.Tensor(100, 300).split([100, 200], dim=1)  # type: ignore[no-untyped-call] # noqa: E501
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.weights
        return x


class SiameseNets(nn.Module):
    """Model with MaxPool and ReLU layers."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)

        self.pooling = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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
    """Model that uses many functional torch layers."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout2d(0.4)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """Model that returns a dict in forward()."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
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
    """Model that uses a ReturnDictLayer."""

    def __init__(self) -> None:
        super().__init__()
        self.return_dict = ReturnDictLayer()

    def forward(self, x: torch.Tensor, y: Any) -> dict[str, torch.Tensor]:
        del y
        activation_dict: dict[str, torch.Tensor] = self.return_dict(x)
        return activation_dict


class DictParameter(nn.Module):
    """Model that takes in a dict in forward()."""

    def __init__(self) -> None:
        super().__init__()
        self.constant = 5

    def forward(self, x: dict[int, torch.Tensor], scale_factor: int) -> torch.Tensor:
        return scale_factor * (x[256] + x[512][0]) * self.constant


class ModuleDictModel(nn.Module):
    """Model that uses a ModuleDict."""

    def __init__(self) -> None:
        super().__init__()
        self.choices = nn.ModuleDict(
            {"conv": nn.Conv2d(10, 10, 3), "pool": nn.MaxPool2d(3)}
        )
        self.activations = nn.ModuleDict({"lrelu": nn.LeakyReLU(), "prelu": nn.PReLU()})

    def forward(
        self, x: torch.Tensor, layer_type: str, activation_type: str
    ) -> torch.Tensor:
        x = self.choices[layer_type](x)
        x = self.activations[activation_type](x)
        return x


class ObjectWithTensors:
    """A class with a 'tensors'-attribute."""

    def __init__(self, tensors: torch.Tensor | Sequence[Any]) -> None:
        self.tensors = tensors


class HighlyNestedDictModel(nn.Module):
    """Model that returns a highly nested dict."""

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 10)

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, tuple[dict[str, list[ObjectWithTensors]]]]:
        x = self.lin1(x)
        x = self.lin2(x)
        x = F.softmax(x, dim=0)
        return {"foo": ({"bar": [ObjectWithTensors(x)]},)}


class IntWithGetitem(int):
    """An int with a __getitem__ method."""

    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self.tensor = tensor

    def __int__(self) -> IntWithGetitem:
        return self

    def __getitem__(self, val: int) -> torch.Tensor:
        return self.tensor * val


class EdgecaseInputOutputModel(nn.Module):
    """
    For testing LayerInfo.calculate_size.extract_tensor:

    case hasattr(inputs, "__getitem__") but not
    isinstance(inputs, (list, tuple, dict)).

    case not inputs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, input_list: dict[str, torch.Tensor]) -> dict[str, IntWithGetitem]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = input_list["foo"] if input_list else torch.ones(3).to(device)
        x = self.linear(x)
        return {"foo": IntWithGetitem(x)}


class NamedTuple(nn.Module):
    """Model that takes in a NamedTuple as input."""

    Point = namedtuple("Point", ["x", "y"])

    def forward(self, x: Any, y: Any, z: Any) -> Any:
        return self.Point(x, y).x + torch.ones(z.x)


class NumpyModel(nn.Module):
    """Model that takes a np.ndarray."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, inp: np.ndarray[Any, Any]) -> Any:
        assert isinstance(inp, np.ndarray)
        x = torch.from_numpy(inp)
        x = self.lin(x)
        return x.cpu().detach().numpy()


class LayerWithRidiculouslyLongNameAndDoesntDoAnything(nn.Module):
    """Model with a very long name."""

    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x: Any) -> Any:
        return self.identity(x)


class EdgeCaseModel(nn.Module):
    """Model that throws an exception when used."""

    def __init__(
        self,
        throw_error: bool = False,
        return_str: bool = False,
        return_class: bool = False,
    ) -> None:
        super().__init__()
        self.throw_error = throw_error
        self.return_str = return_str
        self.return_class = return_class
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.model = LayerWithRidiculouslyLongNameAndDoesntDoAnything()

    def forward(self, x: torch.Tensor) -> Any:
        x = self.conv1(x)
        x = self.model("string output" if self.return_str else x)
        if self.throw_error:
            x = self.conv1(x)
        if self.return_class:
            x = self.model(EdgeCaseModel)
        return x


class PackPaddedLSTM(nn.Module):
    """LSTM model with pack_padded layers."""

    def __init__(
        self,
        vocab_size: int = 60,
        embedding_size: int = 128,
        output_size: int = 18,
        hidden_size: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers=1)
        self.hidden2out = nn.Linear(self.hidden_size, output_size)
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, batch: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden1 = torch.ones(1, batch.size(-1), self.hidden_size, device=batch.device)
        hidden2 = torch.ones(1, batch.size(-1), self.hidden_size, device=batch.device)
        embeds = self.embedding(batch)
        packed_input = pack_padded_sequence(embeds, lengths)
        _, (ht, _) = self.lstm(packed_input, (hidden1, hidden2))
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = F.log_softmax(output, dim=1)
        return cast(torch.Tensor, output)


class ContainerModule(nn.Module):
    """Model using ModuleList."""

    def __init__(self) -> None:
        super().__init__()
        self._layers = nn.ModuleList()
        self._layers.append(nn.Linear(5, 5))
        self._layers.append(ContainerChildModule())
        self._layers.append(nn.Linear(5, 5))
        # Add None, but filter out this value later.
        self._layers.append(None)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self._layers:
            if layer is not None:
                out = layer(out)
        return out


class ContainerChildModule(nn.Module):
    """Model using Sequential in different ways."""

    def __init__(self) -> None:
        super().__init__()
        self._sequential = nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))
        self._between = nn.Linear(5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # call sequential normally, call another layer,
        # loop over sequential without call to forward
        out = self._sequential(x)
        out = self._between(out)
        for layer in self._sequential:
            out = layer(out)

        # call sequential normally, loop over sequential without call to forward
        out = self._sequential(x)
        for layer in self._sequential:
            out = layer(out)
        return cast(torch.Tensor, out)


class EmptyModule(nn.Module):
    """A module that has no layers"""

    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.rand(3, 3, requires_grad=True)
        self.example_input_array = torch.zeros(1, 2, 3, 4, 5)

    def forward(self) -> dict[str, Any]:
        return {"loss": self.parameter.sum()}


class AutoEncoder(nn.Module):
    """Autoencoder module"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.decode = nn.Sequential(nn.Conv2d(16, 3, 3, padding=1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        unpooled_shape = x.size()
        x, indices = self.pool(x)
        # Note: you cannot use keyword argument `input=x` in this function.
        x = self.unpool(x, indices=indices, output_size=unpooled_shape)
        x = self.decode(x)
        return x


class PartialJITModel(nn.Module):
    """Partial JIT model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.fc1 = torch.jit.script(nn.Linear(320, 50))
        self.fc2 = torch.jit.script(nn.Linear(50, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MixedTrainableParameters(nn.Module):
    """Model with trainable and non-trainable parameters in the same layer."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.empty(10), requires_grad=True)
        self.b = nn.Parameter(torch.empty(10), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * x + self.b


class MixedTrainable(nn.Module):
    """Model with fully, partial and non trainable modules."""

    def __init__(self) -> None:
        super().__init__()
        self.fully_trainable = nn.Conv1d(1, 1, 1)

        self.partially_trainable = nn.Conv1d(1, 1, 1, bias=True)
        assert self.partially_trainable.bias is not None
        self.partially_trainable.bias.requires_grad = False

        self.non_trainable = nn.Conv1d(1, 1, 1, 1, bias=True)
        self.non_trainable.weight.requires_grad = False
        assert self.non_trainable.bias is not None
        self.non_trainable.bias.requires_grad = False

        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fully_trainable(x)
        x = self.partially_trainable(x)
        x = self.non_trainable(x)
        x = self.dropout(x)
        return x


class ReuseLinear(nn.Module):
    """Model that uses a reference to the same Linear layer over and over."""

    def __init__(self) -> None:
        super().__init__()
        linear = nn.Linear(10, 10)
        model = []
        for _ in range(4):
            model += [linear, nn.ReLU(True)]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(x))


class ReuseLinearExtended(nn.Module):
    """Model that uses a reference to the same Linear layer over and over."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)
        model = []
        for _ in range(4):
            model += [self.linear, nn.ReLU(True)]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(x))


class ReuseReLU(nn.Module):
    """Model that uses a reference to the same ReLU layer over and over."""

    def __init__(self) -> None:
        super().__init__()
        activation = nn.ReLU(True)
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            activation,
        ]
        for i in range(3):
            mult = 2**i
            model += [
                nn.Conv2d(mult, mult * 2, kernel_size=1, stride=2, padding=1),
                nn.BatchNorm2d(mult * 2),
                activation,
            ]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(x))


class PrunedLayerNameModel(nn.Module):
    """Model that defines parameters with _orig and _mask as suffixes."""

    def __init__(self, input_size: int, attention_size: int) -> None:
        super().__init__()
        self.weight_orig = nn.Parameter(torch.ones((attention_size, input_size)), True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        return self.weight_orig


class FakePrunedLayerModel(nn.Module):
    """Model that defines parameters with _orig and _mask as suffixes."""

    def __init__(self, input_size: int, attention_size: int) -> None:
        super().__init__()
        self.weight_orig = nn.Parameter(torch.ones((attention_size, input_size)), True)
        self.weight_mask = nn.Parameter(torch.zeros((attention_size, input_size)), True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        return self.weight_orig


class RegisterParameter(nn.Sequential):
    """A model with one parameter."""

    weights: list[torch.Tensor]

    def __init__(self, *blocks: nn.Module) -> None:
        super().__init__(*blocks)
        self.register_parameter(
            "weights", nn.Parameter(torch.zeros(len(blocks)).to(torch.float))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for k, block in enumerate(self):
            x += self.weights[k] * block(x)
        return x


class ParameterFCNet(nn.Module):
    """FCNet using Parameters."""

    def __init__(
        self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int | None = None
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.a = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.b = nn.Parameter(torch.randn(hidden_dim))
        if output_dim is not None:
            self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.mm(x, self.a) + self.b
        if self.output_dim is None:
            return h
        return cast(torch.Tensor, self.fc2(h))


class InsideModel(nn.Module):
    """Module with a parameter and an inner module with a Parameter."""

    class Inside(nn.Module):
        """Inner module with a Parameter."""

        def __init__(self) -> None:
            super().__init__()
            self.l_1 = nn.Linear(1, 1)
            self.param_1 = nn.Parameter(torch.ones(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.l_1(x) * self.param_1)

    def __init__(self) -> None:
        super().__init__()
        self.l_0 = nn.Linear(2, 1)
        self.param_0 = nn.Parameter(torch.ones(2))
        self.inside = InsideModel.Inside()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.inside(self.l_0(x)) * self.param_0)


class RecursiveWithMissingLayers(nn.Module):
    """
    Module with more complex recursive layers, which activates add_missing_layers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.out_conv0 = nn.Conv2d(3, 8, 5, padding="same")
        self.out_bn0 = nn.BatchNorm2d(8)

        self.block0 = nn.ModuleDict()
        for i in range(1, 4):
            self.block0.add_module(
                f"in_conv{i}", nn.Conv2d(8, 8, 3, padding="same", dilation=2**i)
            )
            self.block0.add_module(f"in_bn{i}", nn.BatchNorm2d(8))

        self.block1 = nn.ModuleDict()
        for i in range(4, 7):
            self.block1.add_module(
                f"in_conv{i}", nn.Conv2d(8, 8, 3, padding="same", dilation=2 ** (7 - i))
            )
            self.block1.add_module(f"in_bn{i}", nn.BatchNorm2d(8))

        self.out_conv7 = nn.Conv2d(8, 1, 1, padding="same")
        self.out_bn7 = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out_conv0(x)
        x = torch.relu(self.out_bn0(x))

        for i in range(1, 4):
            x = self.block0[f"in_conv{i}"](x)
            x = torch.relu(self.block0[f"in_bn{i}"](x))

        for i in range(4, 7):
            x = self.block1[f"in_conv{i}"](x)
            x = torch.relu(self.block1[f"in_bn{i}"](x))

        x = self.out_conv7(x)
        x = torch.relu(self.out_bn7(x))
        return x


class CNNModuleList(nn.Module):
    """ModuleList with ConvLayers."""

    def __init__(self, conv_layer_cls: type[nn.Module]) -> None:
        super().__init__()
        self.ml = nn.ModuleList([conv_layer_cls() for i in range(5)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.ml:
            x = layer(x)
        return x


class ConvLayerA(nn.Module):
    """ConvLayer with the same module instantiation order in forward()."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)
        out = self.pool(out)
        return cast(torch.Tensor, out)


class ConvLayerB(nn.Module):
    """ConvLayer with a different module instantiation order in forward()."""

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(1, 1, 1)
        self.pool = nn.MaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)
        out = self.pool(out)
        return cast(torch.Tensor, out)


class SimpleRNN(nn.Module):
    """Simple RNN"""

    def __init__(self, repeat_outside_loop: bool = False) -> None:
        super().__init__()
        self.hid_dim = 2
        self.input_dim = 3
        self.max_length = 4
        self.repeat_outside_loop = repeat_outside_loop
        self.lstm = nn.LSTMCell(self.input_dim, self.hid_dim)
        self.activation = nn.Tanh()
        self.projection = nn.Linear(self.hid_dim, self.input_dim)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        b_size = token_embedding.size()[0]
        hx = torch.randn(b_size, self.hid_dim, device=token_embedding.device)
        cx = torch.randn(b_size, self.hid_dim, device=token_embedding.device)

        for _ in range(self.max_length):
            hx, cx = self.lstm(token_embedding, (hx, cx))
            hx = self.activation(hx)

        if self.repeat_outside_loop:
            hx = self.projection(hx)
            hx = self.activation(hx)
        return hx


class MultiDeviceModel(nn.Module):
    """
    A model living on several devices.

    Follows the ToyModel from the Tutorial on parallelism:
    https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html
    """

    def __init__(
        self, device1: torch.device | str, device2: torch.device | str
    ) -> None:
        super().__init__()
        self.device1 = device1
        self.device2 = device2

        self.net1 = torch.nn.Linear(10, 10).to(device1)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(device2)

    def forward(self, x: torch.Tensor) -> Any:
        x = self.relu(self.net1(x.to(self.device1)))
        return self.net2(x.to(self.device2))
