from lightorch.training.supervised import Module
from torch import nn, Tensor
from .loss import criterion


class model_name(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return out


class Model(Module):
    def __init__(self, **hparams) -> None:
        super().__init__(**hparams)
        self.model = model_name()
        self.criterion = criterion()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
