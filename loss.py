from lightorch.nn.criterions import LighTorchLoss
from torch import Tensor
from typing import Callable, Tuple
from scipy.constants import e, proton_mass
import torch
from torchquad import MonteCarlo, set_up_backend

set_up_backend('torch', data_type = 'float32')

class StatisticalMechanicsInformedLoss(LighTorchLoss):
    def __init__(
        self,
        alpha_1: float,
        alpha_2: float,
        alpha_3: float,
        alpha_4: float,
    ) -> None:
        super().__init__(
            labels=[
                "Vlasov",
                "Liouville",
                "Quasyneutrality",
                "Maxwell",
            ],
            factors={
                "Vlasov": alpha_1,
                "Liouville": alpha_2,
                "Quasyneutrality": alpha_3,
                "Maxwell": alpha_4,
            },
        )
        self.integral = MonteCarlo()

    def vlasov(self, f: Tensor, r: Tensor, v: Tensor, t: Tensor, E: Tensor, B: Tensor) -> Tensor:
        """
        This is the solution for the alpha specie. The solution to the overall
        solar wind kinetic model is given by the solution to each specie separately.
        """
        f_r = torch.autograd.grad(
            f, r, create_graph=True
        )[0]
        f_v = torch.autograd.grad(
            f, v, create_graph=True
        )[0]
        f_t = torch.autograd.grad(
            f, t, create_graph=True
        )[0]
        v_cross_B = torch.cross(v, B, dim=-1)

        v_dot_f_r = torch.einsum("bsi,bsi->bs", v, f_r)

        lorentz_m_dot_f_v = torch.einsum(
            "bsi,bsi->bs", (e / proton_mass) * (E + v_cross_B), f_v
        )

        return (f_t + v_dot_f_r + lorentz_m_dot_f_v) ** 2

    def entropy(self, model: Callable) -> Tensor:
        return self.integral.integrate(
            lambda *args: model(*args)*torch.log(model(*args)),
            6,
            10000,
            [
                [0, torch.inf],
                [0, torch.inf],
                [0, torch.inf],
                [0, torch.inf],
                [0, torch.inf],
                [0, torch.inf],
            ]
        )

    def maxwell(self, E_model: Tensor, B_model: Tensor) -> Tensor:

    def forward(self, **kwargs) -> Tuple[Tensor, ...]:
        S = self.entropy(kwargs['model'])

        vlasov = self.vlasov(
            kwargs["f"], kwargs["r"], kwargs["v"], kwargs["t"], kwargs["E"], kwargs["B"]
        )

        dS_dt = torch.autograd.grad(
            S,
            kwargs["t"],
            create_graph = True
        )[0] ** 2

        final = self.factors["PDE"] * vlasov + self.factors["Liouville"] * dS_dt + quasyneutrality * self.factors["Quasyneutrality"] + maxwell * self.factors["Maxwell"]

        return final, vlasov, dS_dt
