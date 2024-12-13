from sympy import Symbol, Function
from scipy.constants import e, m_p, m_e
from modulus.eq.pdes import PDE


class VlasovMaxwell(PDE):
    def __init__(self) -> None:
        super().__init__()
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        t = Symbol("t")
        input = dict(x=x, y=y, z=z, t=t)
        v_x, v_y, v_z = Symbol("v_x"), Symbol("v_y"), Symbol("v_z")

        ## Electric field
        E_x = Function("E_x")(*input)
        E_y = Function("E_y")(*input)
        E_z = Function("E_z")(*input)

        ## Magnetic field
        B_x = Function("B_x")(*input)
        B_y = Function("B_y")(*input)
        B_z = Function("B_z")(*input)

        ## Phase space descriptors for protons and electrons
        f_e = Function(*input, v_x, v_y, v_z)
        f_p = Function(*input, v_x, v_y, v_z)

        ## PDE
        self.equations = dict(
            gauss_elec=E_x.diff(x, 2) + E_y.diff(y, 2) + E_z.diff(z, 2),
            gauss_mag=B_x.diff(x, 2) + B_y.diff(y, 2) + B_z.diff(z, 2),
            faraday_x=E_z.diff(y) - E_y.diff(z) + B_x.diff(t),
            faraday_y=E_z.diff(x) - E_x.diff(z) + B_y.diff(t),
            faraday_z=E_y.diff(x) - E_x.diff(y) + B_z.diff(t),
            vlasov_proton=f_p.diff(t)
            + (v_x * f_p.diff(x) + v_y * f_p.diff(y) + v_z * f_p.diff(z))
            + (e / m_p)
            * (
                (E_x + (v_y * B_z - v_z * B_y)) * f_p.diff(v_x)
                + (E_y + (v_x * B_z - v_z * B_x)) * f_p.diff(v_y)
                + (E_z + (v_x * B_y - v_y * B_x)) * f_p.diff(v_z)
            ),
            vlasov_electron=f_e.diff(t)
            + (v_x * f_e.diff(x) + v_y * f_e.diff(y) + v_z * f_e.diff(z))
            + (-e / m_e)
            * (
                (E_x + (v_y * B_z - v_z * B_y)) * f_e.diff(v_x)
                + (E_y + (v_x * B_z - v_z * B_x)) * f_e.diff(v_y)
                + (E_z + (v_x * B_y - v_y * B_x)) * f_e.diff(v_z)
            ),
            # liouville = ## integral of f_p * log f_p + f_e + log f_e,
            # velocity_constraint = # define
        )
