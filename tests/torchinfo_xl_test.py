import pytest
import torch
import torchvision  # type: ignore[import]

from tests.fixtures.genotype import GenotypeNetwork  # type: ignore[attr-defined]
from tests.fixtures.tmva_net import TMVANet  # type: ignore[attr-defined]
from torchinfo import summary


def test_ascii_only() -> None:
    result = summary(
        torchvision.models.resnet18(),
        depth=3,
        input_size=(1, 3, 64, 64),
        row_settings=["ascii_only"],
    )

    assert str(result).encode("ascii").decode("ascii")


def test_frozen_layers() -> None:
    model = torchvision.models.resnet18()
    for ind, param in enumerate(model.parameters()):
        if ind < 30:
            param.requires_grad = False

    summary(
        model,
        input_size=(1, 3, 64, 64),
        depth=3,
        col_names=("output_size", "num_params", "kernel_size", "mult_adds"),
    )


def test_eval_order_doesnt_matter() -> None:
    input_size = (1, 3, 224, 224)
    input_tensor = torch.ones(input_size)

    model1 = torchvision.models.resnet18(pretrained=True)
    model1.eval()
    summary(model1, input_size=input_size)
    with torch.inference_mode():  # type: ignore[no-untyped-call]
        output1 = model1(input_tensor)

    model2 = torchvision.models.resnet18(pretrained=True)
    summary(model2, input_size=input_size)
    model2.eval()
    with torch.inference_mode():  # type: ignore[no-untyped-call]
        output2 = model2(input_tensor)

    assert torch.all(torch.eq(output1, output2))


def test_resnet18_depth_consistency() -> None:
    model = torchvision.models.resnet18()

    for depth in range(1, 3):
        summary(model, (1, 3, 64, 64), depth=depth, cache_forward_pass=True)


def test_resnet50() -> None:
    # According to https://arxiv.org/abs/1605.07146,
    # resnet50 has ~25.6 M trainable params.
    model = torchvision.models.resnet50()
    results = summary(model, input_size=(2, 3, 224, 224))

    assert results.total_params == 25557032  # close to 25.6e6
    assert results.total_mult_adds == sum(
        layer.macs for layer in results.summary_list if layer.is_leaf_layer
    )


def test_resnet152() -> None:
    model = torchvision.models.resnet152()

    summary(model, (1, 3, 224, 224), depth=3)


@pytest.mark.skip(reason="nondeterministic output")
def test_fasterrcnn() -> None:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained_backbone=False
    )
    results = summary(model, input_size=(1, 3, 112, 112))

    assert results.total_params == 41755286


def test_genotype() -> None:
    model = GenotypeNetwork()

    x = summary(model, (2, 3, 32, 32), depth=3, cache_forward_pass=True)
    y = summary(model, (2, 3, 32, 32), depth=7, cache_forward_pass=True)

    assert x.total_params == y.total_params, (x, y)


def test_tmva_net_column_totals() -> None:
    for depth in (1, 3, 5):
        results = summary(
            TMVANet(n_classes=4, n_frames=5),
            input_data=[
                torch.randn(1, 1, 5, 256, 64),
                torch.randn(1, 1, 5, 256, 256),
                torch.randn(1, 1, 5, 256, 64),
            ],
            col_names=["output_size", "num_params", "mult_adds"],
            depth=depth,
            cache_forward_pass=True,
        )

        assert results.total_params == sum(
            layer.num_params for layer in results.summary_list if layer.is_leaf_layer
        )
        assert results.total_mult_adds == sum(
            layer.macs for layer in results.summary_list if layer.is_leaf_layer
        )


def test_google() -> None:
    summary(torchvision.models.googlenet(), (1, 3, 112, 112), depth=7)
