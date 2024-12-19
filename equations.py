from scipy.constants import e, m_p, m_e, mu_0, epsilon_0
from sympy import Symbol, Function, log
from modulus.eq.pdes import PDE
from typing import Tuple
from torch import Tensor


def define_ind() -> Tuple[Symbol, ...]:
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    t = Symbol("t")
    return x, y, z, t


def define_electric_field(input) -> Tuple[Function, ...]:
    ## Electric field
    E_x = Function("E_x")(*input)
    E_y = Function("E_y")(*input)
    E_z = Function("E_z")(*input)
    return E_x, E_y, E_z


def define_magnetic_field(input) -> Tuple[Function, ...]:
    ## Magnetic field
    B_x = Function("B_x")(*input)
    B_y = Function("B_y")(*input)
    B_z = Function("B_z")(*input)
    return B_x, B_y, B_z


class Maxwell(PDE):
    def __init__(self) -> None:
        super().__init__()
        x, y, z, t = define_ind()
        input = dict(x=x, y=y, z=z, t=t)

        B_x, B_y, B_z = define_magnetic_field(input)
        E_x, E_y, E_z = define_electric_field(input)

        J_x = e * (n_p * v_xp - n_e * v_xe)
        J_y = e * (n_p * v_yp - n_e * v_ye)
        J_x = e * (n_p * v_zp - n_e * v_ze)

        self.equations = dict(
            gauss_elec=E_x.diff(x, 2)
            + E_y.diff(y, 2)
            + E_z.diff(z, 2)
            - (sigma / epsilon_0),
            gauss_mag=B_x.diff(x, 2) + B_y.diff(y, 2) + B_z.diff(z, 2),
            faraday_x=E_z.diff(y) - E_y.diff(z) + B_x.diff(t),
            faraday_y=E_z.diff(x) - E_x.diff(z) + B_y.diff(t),
            faraday_z=E_y.diff(x) - E_x.diff(y) + B_z.diff(t),
            ampere_x=B_z.diff(y) - B_y.diff(z) - mu_0 * (J_x + epsilon_0 * E_x.diff(t)),
            ampere_y=B_z.diff(x) - B_x.diff(z) - mu_0 * (J_y + epsilon_0 * E_y.diff(t)),
            ampere_y=B_y.diff(x) - B_x.diff(y) - mu_0 * (J_z + epsilon_0 * E_z.diff(t)),
        )


class Vlasov(PDE):
    def __init__(self, name: str) -> None:
        super().__init__()
        x, y, z, t = define_ind()
        input = dict(x=x, y=y, z=z, t=t)
        f_p = Function("f_p")(*input)
        B_x, B_y, B_z = define_magnetic_field(input)
        E_x, E_y, E_z = define_electric_field(input)

        self.equations = {
            f"vlasov_{name}": f_p.diff(t)
            + (V_x * f_p.diff(x) + V_y * f_p.diff(y) + V_z * f_p.diff(z))
            + (e / m_p)
            * (
                (E_x + (V_y * B_z - V_z * B_y)) * f_p.diff(V_x)
                + (E_y + (V_x * B_z - V_z * B_x)) * f_p.diff(V_y)
                + (E_z + (V_x * B_y - V_y * B_x)) * f_p.diff(V_z)
            )
        }

    def forward(self, dict_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return


class VlasovProton(PDE):
    def __init__(self) -> None:
        super().__init__()
        x, y, z, t = define_ind()
        input = dict(x=x, y=y, z=z, t=t)
        f_p = Function("f_p")(*input)
        B_x, B_y, B_z = define_magnetic_field(input)
        E_x, E_y, E_z = define_electric_field(input)

        self.equations = dict(
            vlasov_proton=f_p.diff(t)
            + (V_x * f_p.diff(x) + V_y * f_p.diff(y) + V_z * f_p.diff(z))
            + (e / m_p)
            * (
                (E_x + (V_y * B_z - V_z * B_y)) * f_p.diff(V_x)
                + (E_y + (V_x * B_z - V_z * B_x)) * f_p.diff(V_y)
                + (E_z + (V_x * B_y - V_y * B_x)) * f_p.diff(V_z)
            )
        )


class VlasovElectron(PDE):
    def __init__(self) -> None:
        super().__init__()
        x, y, z, t = define_ind()
        input = dict(x=x, y=y, z=z, t=t)
        f_e = Function("f_e")(*input)
        B_x, B_y, B_z = define_magnetic_field(input)
        E_x, E_y, E_z = define_electric_field(input)
        V_x, V_y, V_z = define_velocity_field(input)

        self.equations = dict(
            vlasov_electron=f_e.diff(t)
            + (V_x * f_e.diff(x) + V_y * f_e.diff(y) + V_z * f_e.diff(z))
            + (e / m_p)
            * (
                (E_x + (V_y * B_z - V_z * B_y)) * f_e.diff(V_x)
                + (E_y + (V_x * B_z - V_z * B_x)) * f_e.diff(V_y)
                + (E_z + (V_x * B_y - V_y * B_x)) * f_e.diff(V_z)
            )
        )


class EnergyConservation(PDE):
    def __init__(self) -> None:
        super().__init__()
        x, y, z, t = define_ind()
        B_x, B_y, B_z = define_magnetic_field(input)
        E_x, E_y, E_z = define_electric_field(input)
        V_x, V_y, V_z = define_velocity_field(input)

        self.equations = dict(
            energy_conservation=(
                (1 / 2) * (m_e * m_p) * (V_x**2 + V_y**2 + V_z**2)
            ).diff(t)
            + (1 / mu_0)
            * (
                (E_y * B_z - E_z * B_y).diff(x)
                + (E_x * B_z - E_z * B_x).diff(y)
                + (E_x * B_y - E_y * B_x).diff(z)
            )
        )


class Liouville(PDE):
    def __init__(self, name: str) -> None:
        super().__init__()
        x, y, z, t = define_ind()
        V_x, V_y, V_z = define_velocity_field(input)
        f_e = Function("f_e")(*input)
        self.equations = dict(
            liouville_electron=(
                (f_e * log(f_e))
                .integrate(x)
                .integrate(y)
                .integrate(z)
                .integrate(V_x)
                .integrate(V_y)
                .integrate(V_z)
            ).diff(t)
        )
