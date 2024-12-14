from sympy import Symbol, Function, log
from scipy.constants import e, m_p, m_e, mu_0
from modulus.eq.pdes import PDE
from itertools import chain
from typing import List, Tuple

def get_liouville(f_alpha: Function, cfg: ModulusConfig, *input):
    integrand = f_alpha * log(f_alpha)
    bounds: List[Tuple[float, float]] = [*chain.from_iterable(cfg.bounds)]
    integral: int = 0
    integral = integrand.integrate(input[0], bounds[0])
    for x_n, bound in zip(input[1:], bounds[1:]):
        integral = integrand.integrate(x_n, bound)
    return integral

class VlasovMaxwell(PDE):
    def __init__(self, cfg: ModulusConfig) -> None:
        super().__init__()
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        t = Symbol("t")
        input = dict(x=x, y=y, z=z, t=t)
        ## Electric field
        E_x = Function("E_x")(*input)
        E_y = Function("E_y")(*input)
        E_z = Function("E_z")(*input)

        ## Magnetic field
        B_x = Function("B_x")(*input)
        B_y = Function("B_y")(*input)
        B_z = Function("B_z")(*input)

        ## Velocity field
        V_x = Function("V_x")(*input)
        V_y = Function("V_y")(*input)
        V_z = Function("V_z")(*input)

        ## Phase space descriptors for protons and electrons
        f_e = Function(*input, V_x, V_y, V_z)
        f_p = Function(*input, V_x, V_y, V_z)

        ## PDE
        self.equations = dict(
            gauss_elec=E_x.diff(x, 2) + E_y.diff(y, 2) + E_z.diff(z, 2),
            gauss_mag=B_x.diff(x, 2) + B_y.diff(y, 2) + B_z.diff(z, 2),
            faraday_x=E_z.diff(y) - E_y.diff(z) + B_x.diff(t),
            faraday_y=E_z.diff(x) - E_x.diff(z) + B_y.diff(t),
            faraday_z=E_y.diff(x) - E_x.diff(y) + B_z.diff(t),
            vlasov_proton=f_p.diff(t)
                + (V_x * f_p.diff(x) + V_y * f_p.diff(y) + V_z * f_p.diff(z))
                + (e / m_p)
                    * (
                        (E_x + (V_y * B_z - V_z * B_y)) * f_p.diff(V_x)
                            + (E_y + (V_x * B_z - V_z * B_x)) * f_p.diff(V_y)
                            + (E_z + (V_x * B_y - V_y * B_x)) * f_p.diff(V_z)
                    ),
            vlasov_electron=f_e.diff(t)
                + (V_x * f_e.diff(x) + V_y * f_e.diff(y) + V_z * f_e.diff(z))
                + (-e / m_e)
                    * (
                        (E_x + (V_y * B_z - V_z * B_y)) * f_e.diff(V_x)
                            + (E_y + (V_x * B_z - V_z * B_x)) * f_e.diff(V_y)
                            + (E_z + (V_x * B_y - V_y * B_x)) * f_e.diff(V_z)
                    ),
            liouville = get_liouville(f_p, cfg, *input) + get_liouville(f_e, cfg, *input),
            energy = ( (1/2) * (m_e * m_p) * (V_x**2 + V_y**2 + V_z**2)).diff(t) + (1/mu_0) * (
                (E_y * B_z - E_z * B_y).diff(x) + (E_x * B_z - E_z * B_x).diff(y) + (E_x * B_y - E_y * B_x).diff(z)
            )
        )
