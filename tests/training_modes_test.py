import pytest
import torch
from torch import nn

from torchinfo import summary


class MixedTrainingModel(nn.Module):
    def __init__(self, *, training: bool, fail: bool) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 3), nn.Dropout())
        self.fail = fail
        self.observed_modes: list[bool] = []
        self.train(training)
        self.layers[0].train(not training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.observed_modes = [module.training for module in self.modules()]
        if self.fail:
            raise ValueError("forward failed")
        return self.layers(x)  # type: ignore[no-any-return]


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("mode", ["same", "train", "eval"])
@pytest.mark.parametrize("fail", [True, False])
def test_preserve_module_training_modes(
    training: bool, mode: str, *, fail: bool
) -> None:
    model = MixedTrainingModel(training=training, fail=fail)
    before = [module.training for module in model.modules()]
    if fail:
        with pytest.raises(RuntimeError, match="Failed to run torchinfo") as caught:
            summary(model, input_data=torch.zeros(2, 3), mode=mode, verbose=0)
        assert isinstance(caught.value.__cause__, ValueError)
    else:
        summary(model, input_data=torch.zeros(2, 3), mode=mode, verbose=0)
    expected = before if mode == "same" else [mode == "train"] * len(before)
    assert model.observed_modes == expected
    assert [module.training for module in model.modules()] == before
    assert all(not module._forward_hooks for module in model.modules())
    assert all(not module._forward_pre_hooks for module in model.modules())


def test_preserve_shared_module_training_modes() -> None:
    shared = nn.Linear(3, 3)
    model = nn.Sequential(nn.Sequential(shared), nn.Sequential(shared))
    model.train()
    model[0].eval()
    model[1].train()
    before = [module.training for module in model.modules()]
    summary(model, input_data=torch.zeros(2, 3), mode="eval", verbose=0)
    assert [module.training for module in model.modules()] == before
