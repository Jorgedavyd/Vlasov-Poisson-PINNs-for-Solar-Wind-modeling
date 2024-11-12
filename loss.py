from lightorch.nn.criterions import LighTorchLoss
from torch import Tensor
from typing import Tuple
from scipy.constants import e, proton_mass
import torch


class StatisticalMechanicsInformedLoss(LighTorchLoss):
    def __init__(
        self,
        alpha_1: float,
        alpha_2: float,
    ) -> None:
        super().__init__(
            labels=[
                "PDE",
                "CurrentDensityCondition",
                "LiouvilleCondition",
                "ProbabilisticCondition",
                "Entropy",
            ],
            factors={
                "PDE": alpha_1,
                "BoundaryConditions": alpha_2,
            },
        )

    def forward(self, **kwargs) -> Tuple[Tensor, ...]:
        f_alpha_r = torch.autograd.grad(
            kwargs["f_alpha"], kwargs["r"], create_graph=True
        )[0]
        f_alpha_v = torch.autograd.grad(
            kwargs["f_alpha"], kwargs["v"], create_graph=True
        )[0]
        f_alpha_t = torch.autograd.grad(
            kwargs["f_alpha"], kwargs["t"], create_graph=True
        )[0]
        v_cross_B = torch.cross(kwargs["v"], kwargs["B"], dim=-1)

        v_dot_f_alpha_r = torch.einsum("bsi,bsi->bs", kwargs["v"], f_alpha_r)

        F_m_dot_f_alpha_v = torch.einsum(
            "bsi,bsi->bs", (e / proton_mass) * (kwargs["E"] + v_cross_B), f_alpha_v
        )

        vlasov = (f_alpha_t + v_dot_f_alpha_r + F_m_dot_f_alpha_v) ** 2

        integral_rv = torch.sum(kwargs["f_alpha"]) * self.volume_phase
        j_density = torch.sum(kwargs["f_alpha"] * kwargs["v"]) * self.volume_phase

        current_density = e * j_density
        liouville = (
            torch.autograd.grad(integral_rv, kwargs["t"], create_graph=True)[0] ** 2
        )

        non_reduced_entropy = -kwargs["f_alpha"] * torch.log(kwargs["f_alpha"])

        entropy_criteria = -torch.autograd.grad(
            torch.sum(non_reduced_entropy) * self.volume_phase,
            kwargs["t"],
            create_graph=True,
        )[0]

        prob = (integral_rv - 1) ** 2
        loss = vlasov * self.factors["PDE"] + self.factors["BoundaryConditions"] * (
            liouville + prob + current_density + entropy_criteria
        )
        return loss, vlasov, current_density, liouville, prob, entropy_criteria
