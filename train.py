from sympy import Symbol
import modulus
from modulus.hydra import ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from model import NeuralNetwork
from equations import VlasovMaxwell
from modulus.key import Key

from modulus.geometry.parametrization import Parametrization, Parameter
from modulus.geometry.primitives_3d import Box, Sphere, Cylinder, Plane
from modulus.geometry.primitives_2d import Rectangle, Circle, Line
from modulus.geometry.primitives_1d import Point1D, Line1D
from modulus.utils.io.vtk import var_to_polyvtk


def make_geometry():
    return ...


@modulus.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    t = Symbol("t")
    input = dict(x=x, y=y, z=z, t=t)
    v_x, v_y, v_z = Symbol("v_x"), Symbol("v_y"), Symbol("v_z")

    eq = VlasovMaxwell()
    full_model = NeuralNetwork(cfg)
    nodes = eq.make_nodes() + [full_model.make_node(name="neural_network")]
    geo = make_geometry()
    domain = Domain()

    ## define L1 boundaries
    boundary = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo, outvar={"E": 0}, batch_size=2
    )

    ## residual criteria
    residual = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar=dict(
            gauss_elec=0,
            gauss_mag=0,
            faraday_x=0,
            faraday_y=0,
            faraday_z=0,
            vlasov_proton=0,
            vlasov_electron=0,
        ),
        batch_size=100,
        bounds=dict(r=(0, 210), t=(0, 7200)),
    )

    integral = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"VlasovMaxwell": 0},
        batch_size=1,
        integral_batch_size=1000,
    )  ## monte carlo integration for this one

    domain.add_constraint(integral, "dirichlet")
    domain.add_constraint(residual, "residual")

    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()
