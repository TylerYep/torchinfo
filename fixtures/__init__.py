"""Test fixures."""
from .genotype import GenotypeNetwork  # type: ignore[attr-defined]
from .models import (
    AutoEncoder,
    ContainerModule,
    CustomParameter,
    EdgeCaseModel,
    EmptyModule,
    LinearModel,
    LSTMNet,
    ModuleDictModel,
    MultipleInputNetDifferentDtypes,
    NamedTuple,
    PackPaddedLSTM,
    ParameterListModel,
    PartialJITModel,
    RecursiveNet,
    ReturnDict,
    SiameseNets,
    SingleInputNet,
)
from .tmva_net import TMVANet  # type: ignore[attr-defined]

__all__ = (
    "AutoEncoder",
    "ContainerModule",
    "CustomParameter",
    "EdgeCaseModel",
    "EmptyModule",
    "GenotypeNetwork",
    "LSTMNet",
    "LinearModel",
    "ModuleDictModel",
    "MultipleInputNetDifferentDtypes",
    "NamedTuple",
    "PackPaddedLSTM",
    "ParameterListModel",
    "PartialJITModel",
    "RecursiveNet",
    "ReturnDict",
    "SiameseNets",
    "SingleInputNet",
    "TMVANet",
)
