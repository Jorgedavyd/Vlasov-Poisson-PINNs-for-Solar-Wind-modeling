from lightorch.nn.criterions import LighTorchLoss
from torch import Tensor


class criterion(LighTorchLoss):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            labels=[],
            factors=[],
        )

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
