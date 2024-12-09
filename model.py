from lightorch.training.supervised import Module
from lightorch.nn import DeepNeuralNetwork
from torch import nn, Tensor
from .loss import StatisticalMechanicsInformedLoss
from typing import Sequence, Union


class PINN(DeepNeuralNetwork):
    def __init__(
        self,
        in_features: int,
        layers: Sequence[int],
        activations: Sequence[Union[str, None]],
    ):
        super().__init__(
            in_features,
            layers,
            list(
                map(
                    lambda activation: (
                        getattr(nn, activation) if activation is not None else None
                    ),
                    activations,
                )
            ),
        )

    def forward(self, F: Tensor) -> Tensor:
        return super().forward()


class Model(Module):
    def __init__(self, **hparams) -> None:
        super().__init__(**hparams)
        self.model = PINN(3, hparams["layers"], hparams["activations"])
        self.criterion = StatisticalMechanicsInformedLoss(
            hparams["alpha_1"], hparams["alpha_2"], hparams["alpha_3"],
            PINN(), PINN(), PINN(), PINN()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
