from sympy import Symbol
from argparse import ArgumentParser
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

from modulus.geometry.parametrization import Parametrization, Parameter
from modulus.geometry.primitives_3d import Box, Sphere, Cylinder, Plane
from modulus.geometry.primitives_2d import Rectangle, Circle, Line
from modulus.geometry.primitives_1d import Point1D, Line1D
from modulus.utils.io.vtk import var_to_polyvtk


def make_geometry(cfg: ModulusConfig):
    nr_points: int = cfg.geometry.grid_resolution
    ball = Sphere(center = (0, 0, 0), radius = cfg.geometry.max_length)
    s = ball.sample_boundary(nr_points = nr_points)
    var_to_polyvtk(s, "general_geometry")
    return s

def define_constraints(cfg, nodes):
    geo = make_geometry(cfg)
    domain = Domain()

    ## define L1 boundaries
    boundary_1 = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo, outvar={"E": 0}, batch_size=cfg.batch_size.boundary_1
    )
    ## define other dirichlet
    boundary_2 = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo, outvar={"E": 0}, batch_size=cfg.batch_size.boundary_1
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
            liouville = 0,
            energy = 0,
        ),
        batch_size=cfg.batch_size.pde,
        bounds=dict(r=cfg.bounds.r, t=cfg.bounds.t),
    )

    domain.add_constraint(boundary_1, "dirichlet_1")
    domain.add_constraint(boundary_2, "dirichlet_2")
    domain.add_constraint(residual, "residual")
    return domain

@modulus.main(version_base="1.3", config_path="conf", config_name="config")
def train_nn(cfg: ModulusConfig) -> None:
    eq = VlasovMaxwell(cfg)
    full_model = NeuralNetwork(cfg)
    nodes = eq.make_nodes() + [full_model.make_node(name="neural_network")]
    domain = define_constraints(cfg, nodes)
    slv = Solver(cfg, domain)
    slv.solve()


@modulus.main(version_base="1.3", config_path="conf", config_name="config")
def train_fno(cfg: ModulusConfig) -> None:
    eq = VlasovMaxwell(cfg)
    full_model = NeuralNetwork(cfg) ## define fno
    nodes = eq.make_nodes() + [full_model.make_node(name="neural_network")]
    domain = define_constraints(cfg, nodes)
    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    parser = ArgumentParser(prog = "VlasovMaxwell-PINN", description = "Statistical mechanics informed Neural Network training for solar wind modeling")
    parser.add_argument("-t", "--type", help = "Define the architecture of the models")
    args = parser.parse_args()
    if args.type == "nn":
        train_nn()
    elif args.type == "fno":
        train_fno()
