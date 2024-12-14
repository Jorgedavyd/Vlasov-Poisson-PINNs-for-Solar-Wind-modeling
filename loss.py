from lightorch.nn.criterions import LighTorchLoss
from torch import Tensor
from typing import Callable, Tuple, Dict
from scipy.constants import e, m_e, m_p
import torch
from torchquad import MonteCarlo, set_up_backend
from modulus.loss import Loss

set_up_backend("torch", data_type="float32")

class LiouvilleLoss(Loss):
    def __init__(self, cfg: ModulusConfig) -> None:
        super().__init__()
        self.integral_method = MonteCarlo()
        self.dimensionality: int = cfg.dimensions * 2 + 1
        self.sampling = cfg.sampling_monte_carlo
        self.bounds = cfg.bounds

    def forward(self, f_alpha: Callable) -> Dict[str, Tensor]:
        entropy = self.integral.integrate(
            lambda *args: f_alpha(*args) * torch.log(f_alpha(*args)),
            self.dimensionality,
            self.sampling,
            [
                *self.bounds.r,
                *self.bounds.v,
                self.bounds.t
            ],
        )

        dS_dt = torch.autograd.grad(
            inputs = t,
            outputs = entropy,
            grad_outputs = torch.ones_like(entropy),
        )[0]

        return {"Liouville": dS_dt}


class StatisticalMechanicsInformedLoss(LighTorchLoss):
    def __init__(
        self,
        alpha_1: float,
        alpha_2: float,
        alpha_3: float,
        f_E: Callable,
        f_B: Callable,
        f_e: Callable,
        f_p: Callable,
    ) -> None:
        super().__init__(
            labels=[
                "Vlasov",
                "Liouville",
                "Maxwell",
            ],
            factors={
                "Vlasov": alpha_1,
                "Liouville": alpha_2,
                "Maxwell": alpha_3,
            },
        )

        self.integral = MonteCarlo()
        self.f_E = f_E
        self.f_B = f_B
        self.f_e = f_e
        self.f_p = f_p

    def boundary(self, E_L1: Tensor, B_L1: Tensor) -> Tensor:
        f_alpha_v_min = 0.0
        f_alpha_v_max = 0.0
        f_alpha_t_min = 0.0
        f_alpha_t_max = 0.0
        f_E_r_min = E
        f_B_r_min = B
        return  ## revisar como hacer el dataset y demas cosas

    def vlasov(
        self,
        f_alpha: Tensor,
        input: Tensor,
        v: Tensor,
        E: Tensor,
        B: Tensor,
        m_alpha: float,
        q_alpha: float,
    ) -> Tensor:
        """
        This is the solution f_alphaor the alpha specie. The solution to the overall
        solar wind kinetic model is given by the solution to each specie separately.
        """

        grad_f_alpha = torch.autograd.grad(
            inputs=input,
            outputs=f_alpha,
            grad_outputs=torch.ones_like(f_alpha),
            retain_graph=True,
            create_graph=True,
        )[0]

        f_alpha_r, f_alpha_v, f_alpha_t = (
            grad_f_alpha[:, :, :3],
            grad_f_alpha[:, :, 3:6],
            grad_f_alpha[:, :, -1],
        )  ## 3-d

        v_cross_B = torch.cross(v, B, dim=-1)

        v_dot_f_alpha_r = torch.einsum("bsi,bsi->bs", v, f_alpha_r)

        lorentz_m_dot_f_alpha_v = torch.einsum(
            "bsi,bsi->bs", (q_alpha / m_alpha) * (E + v_cross_B), f_alpha_v
        )

        return ((f_alpha_t + v_dot_f_alpha_r + lorentz_m_dot_f_alpha_v) ** 2).sum()

    def entropy(self, model: Callable) -> Tensor:
        return self.integral.integrate(
            lambda *args: model(*args) * torch.log(model(*args)),
            6,
            10000,
            [
                [0, 230],
                [0, 1200],
                [0, 7200],
            ],
        )

    def maxwell(self, E: Tensor, B: Tensor, input: Tensor) -> Tensor:
        Ex, Ey, Ez = E[:, :, 0], E[:, :, 1], E[:, :, 2]
        Bx, By, Bz = B[:, :, 0], B[:, :, 1], B[:, :, 2]

        grad_Ex = torch.autograd.grad(
            inputs=input,
            outputs=Ex,
            grad_outputs=torch.ones_like(Ex),
            retain_graph=True,
            create_graph=True,
        )[0]

        grad_Ey = torch.autograd.grad(
            inputs=input,
            outputs=Ey,
            grad_outputs=torch.ones_like(Ey),
            retain_graph=True,
            create_graph=True,
        )[0]

        grad_Ez = torch.autograd.grad(
            inputs=input,
            outputs=Ez,
            grad_outputs=torch.ones_like(Ez),
            retain_graph=True,
            create_graph=True,
        )[0]

        grad_Bx = torch.autograd.grad(
            inputs=input,
            outputs=Bx,
            grad_outputs=torch.ones_like(Bx),
            retain_graph=True,
            create_graph=True,
        )[0]

        grad_By = torch.autograd.grad(
            inputs=input,
            outputs=By,
            grad_outputs=torch.ones_like(By),
            retain_graph=True,
            create_graph=True,
        )[0]

        grad_Bz = torch.autograd.grad(
            inputs=input,
            outputs=Bz,
            grad_outputs=torch.ones_like(Bz),
            retain_graph=True,
            create_graph=True,
        )[0]

        ## Quasineutral Gauss-Ostrogradski-Poisson (electric field)
        ## div(E) = 0
        ## min_theta div(E(r, t; theta))^2
        quasineutral_poisson = (
            grad_Ex[:, :, 0] + grad_Ey[:, :, 1] + grad_Ez[:, :, 2]
        ) ** 2

        ## Gauss-Ostrogradski for (magnetic field)
        magnetic_gauss = (grad_Bx[:, :, 0] + grad_By[:, :, 1] + grad_Bz[:, :, 2]) ** 2

        ## Faraday
        nabla_cross_Ex = grad_Ez[:, :, 1] - grad_Ey[:, :, 2]
        nabla_cross_Ey = grad_Ez[:, :, 0] - grad_Ex[:, :, 2]
        nabla_cross_Ez = grad_Ey[:, :, 0] - grad_Ex[:, :, 1]

        faraday_x = (nabla_cross_Ex + grad_Bx[:, :, 3]) ** 2
        faraday_y = (nabla_cross_Ey + grad_By[:, :, 3]) ** 2
        faraday_z = (nabla_cross_Ez + grad_Bz[:, :, 3]) ** 2

        faraday = faraday_x + faraday_y + faraday_z

        return quasineutral_poisson + magnetic_gauss + faraday

    def forward(self, **kwargs) -> Tuple[Tensor, ...]:
        ### Forward pass of models
        E: Tensor = self.f_E(kwargs["input"])
        B: Tensor = self.f_B(kwargs["input"])
        f_e: Tensor = self.f_e(kwargs["input"], v, E, B, m_e, -e)
        f_p: Tensor = self.f_p(kwargs["input"], v, E, B, m_p, e)

        maxwell = self.maxwell(E, B, kwargs["input"])

        vlasov = self.vlasov(f_e, kwargs["inputs"], v, E, B, m_e, -e) + self.vlasov(
            f_p, kwargs["input"], v, E, B, m_p, e
        )

        S = self.entropy(self.f_e) + self.entropy(self.f_p)
        dS_dt = (
            torch.autograd.grad(
                inputs=kwargs["input"][:, -1],
                outputs=S,
                grad_outputs=torch.ones_like(S),
                retain_graph=True,
                create_graph=True,
            )[0]
            ** 2
        )
        bc = self.boundary(*kwargs["bc"])

        final = (
            self.factors["Vlasov"] * vlasov
            + self.factors["Liouville"] * dS_dt
            + maxwell * self.factors["Maxwell"]
            + self.factors["BC"] * bc
        )

        return final, vlasov, dS_dt, maxwell
