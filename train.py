from modulus.sym.hydra import ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from sympy import Symbol, Function
from argparse import ArgumentParser
from typing import Dict
from torch.utils.data import Dataset
from dataset import L1Dataset
import modulus
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    SupervisedGridConstraint
)
from equations import VlasovMaxwell
from model import NeuralNetwork

from modulus.sym.geometry.primitives_3d import Sphere
from modulus.sym.utils.io.vtk import var_to_polyvtk


def make_geometry(cfg: ModulusConfig):
    nr_points: int = cfg.geometry.grid_resolution
    ball = Sphere(center = (0, 0, 0), radius = cfg.geometry.max_length)
    s = ball.sample_boundary(nr_points = nr_points)
    var_to_polyvtk(s, "general_geometry")
    return s

def define_constraints(cfg, nodes):
    x, y, z, t = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("t")
    input: Dict[str, Symbol] = dict(
        x = x, y = y, z = z, t = t
    )
    ## Velocity field
    V_x = Function("V_x")(*input)
    V_y = Function("V_y")(*input)
    V_z = Function("V_z")(*input)
    geo = make_geometry(cfg)
    domain = Domain()
    path: str = "/data/Vlasov/curated.csv"
    train_dataset: Dataset = L1Dataset(path, [
        "B_x", "B_y", "B_z",
        "E_x", "E_y", "E_z",
        "V_x", "V_y", "V_z",
    ])

    ## define L1 boundaries
    boundary_1 = SupervisedGridConstraint(
        nodes=nodes, geometry=geo, dataset = train_dataset,
        batch_size = cfg.batch_size.boundary1,
        parametrization = {t:0}
    )

    ## define initial conditions for the f_e and f_p
    boundary_2 = PointwiseBoundaryConstraint(
        nodes = nodes, geometry = geo,
        outvar = dict(f_e = 0., f_p = 0.), ## setup initial coniditons
        parametrization = {t:0}
    )

    # velocity extreme cases
    vx_boundary = PointwiseBoundaryConstraint(
        nodes = nodes, geometry = geo,
        outvar = dict(f_e = 0., f_p = 0.),
        parametrization = {V_x: cfg.bounds[0]}
    )

    # velocity extreme cases
    vy_boundary = PointwiseBoundaryConstraint(
        nodes = nodes, geometry = geo,
        outvar = dict(f_e = 0., f_p = 0.),
        parametrization = {V_y: cfg.bounds[1]}
    )

    # velocity extreme cases
    vz_boundary = PointwiseBoundaryConstraint(
        nodes = nodes, geometry = geo,
        outvar = dict(f_e = 0., f_p = 0.),
        parametrization = {V_z: cfg.bounds[1]}
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
        bounds=dict(r=cfg.bounds.r, v=cfg.bounds.v, t=cfg.bounds.t),
    )

    domain.add_constraint(boundary_1, "dirichlet_1")
    domain.add_constraint(boundary_2, "dirichlet_2")
    domain.add_constraint(vx_boundary, "vx_boundary")
    domain.add_constraint(vy_boundary, "vy_boundary")
    domain.add_constraint(vz_boundary, "vz_boundary")
    domain.add_constraint(residual, "residual")

    return domain

@modulus.sym.main(version_base="1.3", config_path="conf", config_name="config")
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
