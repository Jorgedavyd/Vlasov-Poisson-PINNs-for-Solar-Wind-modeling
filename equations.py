from scipy.constants import e, m_p, m_e, mu_0, epsilon_0
from sympy import Symbol, Function
from modulus.sym.eq.pdes import PDE
from modulus.sym.geometry import Geometry
from typing import Tuple, List
from torch import Tensor
from torch import nn, Tensor
import torch
from torchquad import MonteCarlo, set_up_backend

set_up_backend('torch', data_type=torch.float32)

def define_ind() -> Tuple[Symbol, ...]:
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    t = Symbol("t")
    return x, y, z, t

def define_velocity(input, specie: str) -> Tuple[Function, ...]:
    if specie == 'p':
        return Function("v_xp")(*input), Function("v_yp")(*input), Function("v_zp")(*input)
    elif specie == 'e':
        return Function("v_xe")(*input), Function("v_ye")(*input), Function("v_ze")(*input)
    else:
        raise ValueError("Not valid specie")

def define_electric_field(input) -> Tuple[Function, ...]:
    E_x = Function("E_x")(*input)
    E_y = Function("E_y")(*input)
    E_z = Function("E_z")(*input)
    return E_x, E_y, E_z

def define_magnetic_field(input) -> Tuple[Function, ...]:
    B_x = Function("B_x")(*input)
    B_y = Function("B_y")(*input)
    B_z = Function("B_z")(*input)
    return B_x, B_y, B_z

class Density(nn.Module):
    def __init__(self, specie: str, geometry: Geometry, nr_points: int, montecarlo_points: int) -> None:
        super().__init__()
        self.specie = specie
        self.integral = MonteCarlo()
        self.geometry = geometry
        self.nr_points = nr_points
        self.monte_points = montecarlo_points

    def forward(self, f_alpha: nn.Module, t: float | int) -> Tensor:
        points = self.geometry.sample_interior(nr_points = self.nr_points)
        output: List[Tensor] | Tensor = []
        def integrand(rx, ry, rz, vx, vy, vz, t) -> Tensor:
            output = f_alpha({
                "x": rx,
                "y": ry,
                "z": rz,
                f"v_x{self.specie}": vx,
                f"v_y{self.specie}": vy,
                f"v_z{self.specie}": vz,
                "t": t
            })
            return output[f"f_{self.specie}"]

        for rx, ry, rz in points:
            rho: Tensor = self.integral.integrate(
                fn = lambda vx, vy, vz: integrand(rx, ry, rz, vx, vy, vz, t),
                N = self.monte_points,
                integration_domain = [], ## define with geometry
                backend = 'torch',
                dim = -1,
            )
            output.append(rho)
        return torch.stack(output)

class Velocity(nn.Module):
    def __init__(self, specie: str, geometry: Geometry, nr_points: int, montecarlo_points: int) -> None:
        super().__init__()
        self.specie = specie
        self.integral = MonteCarlo()
        self.geometry = geometry
        self.nr_points = nr_points
        self.monte_points = montecarlo_points

    def forward(self, f_alpha: nn.Module, t: float | int, rho: Tensor) -> Tensor:
        points = self.geometry.sample_interior(nr_points = self.nr_points)
        output: List[Tensor] | Tensor = []
        def integrand(rx, ry, rz, vx, vy, vz, t) -> Tensor:
            output = f_alpha({
                "x": rx,
                "y": ry,
                "z": rz,
                f"v_x{self.specie}": vx,
                f"v_y{self.specie}": vy,
                f"v_z{self.specie}": vz,
                "t": t
            })
            value = output[f"f_{self.specie}"]
            return torch.cat([value * vx, value * vy, value * vz], dim = -1)

        for rx, ry, rz in points:
            v: Tensor = self.integral.integrate(
                fn = lambda vx, vy, vz: integrand(rx, ry, rz, vx, vy, vz, t),
                N = self.monte_points,
                integration_domain = [], ## define with geometry
                backend = 'torch',
                dim = -1,
            )
            output.append(v)
        return torch.stack(output) / rho


class Maxwell(PDE):
    def __init__(self) -> None:
        super().__init__()
        x, y, z, t = define_ind()
        input = dict(x=x, y=y, z=z, t=t)

        B_x, B_y, B_z = define_magnetic_field(input)
        E_x, E_y, E_z = define_electric_field(input)
        rho_p, rho_e = Function("rho_p")(*input), Function("rho_e")(*input)
        v_xp, v_yp, v_zp = define_velocity(input, "p")
        v_xe, v_ye, v_ze = define_velocity(input, "e")

        J_x = e * (rho_p * v_xp - rho_e * v_xe)
        J_y = e * (rho_p * v_yp - rho_e * v_ye)
        J_z = e * (rho_p * v_zp - rho_e * v_ze)

        self.equations = dict(
            gauss_elec=E_x.diff(x, 2) + E_y.diff(y, 2) + E_z.diff(z, 2),
            gauss_mag=B_x.diff(x, 2) + B_y.diff(y, 2) + B_z.diff(z, 2),
            faraday_x=E_z.diff(y) - E_y.diff(z) + B_x.diff(t),
            faraday_y=E_z.diff(x) - E_x.diff(z) + B_y.diff(t),
            faraday_z=E_y.diff(x) - E_x.diff(y) + B_z.diff(t),
            ampere_x=B_z.diff(y) - B_y.diff(z) - mu_0 * (J_x + epsilon_0 * E_x.diff(t)),
            ampere_y=B_z.diff(x) - B_x.diff(z) - mu_0 * (J_y + epsilon_0 * E_y.diff(t)),
            ampere_z=B_y.diff(x) - B_x.diff(y) - mu_0 * (J_z + epsilon_0 * E_z.diff(t)),
        )

class Vlasov(PDE):
    def __init__(self, specie: str) -> None:
        super().__init__()
        if specie == 'p':
            m_alpha: float = m_p
            q_alpha: float = e
        elif specie == 'e':
            m_alpha: float = m_e
            q_alpha: float = -e
        else:
            raise ValueError("Not valid specie")

        x, y, z, t = define_ind()
        input = dict(x=x, y=y, z=z, t=t)
        f_alpha = Function(f"f_{specie}")(*input)
        B_x, B_y, B_z = define_magnetic_field(input)
        E_x, E_y, E_z = define_electric_field(input)
        vx, vy, vz = define_velocity(input, specie)

        self.equations = {
            f"vlasov_{specie}": f_alpha.diff(t)
            + (vx * f_alpha.diff(x) + vy * f_alpha.diff(y) + vz * f_alpha.diff(z))
            + (q_alpha / m_alpha)
            * (
                (E_x + (vy * B_z - vz * B_y)) * f_alpha.diff(vx)
                + (E_y + (vx * B_z - vz * B_x)) * f_alpha.diff(vy)
                + (E_z + (vx * B_y - vy * B_x)) * f_alpha.diff(vz)
            )
        }

class Continuity(PDE):
    def __init__(self, specie: str) -> None:
        super().__init__()
        x, y, z, t = define_ind()
        input = dict(x=x, y=y, z=z, t=t)
        rho = Function(f"rho_{specie}")(*input)
        vx, vy, vz = define_velocity(input, specie)
        self.equations = {
            f"continuity_{specie}": rho.diff(t) + (vx * rho).diff(x) + (vy * rho).diff(y) + (vz * rho).diff(z),
        }
