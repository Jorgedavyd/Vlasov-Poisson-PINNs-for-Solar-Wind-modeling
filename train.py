from modulus.sym.hydra import ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from sympy import Or
from argparse import ArgumentParser
from typing import List, Callable
import modulus.sym
from modulus.sym import Node
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    SupervisedGridConstraint,
)
from equations import (
    EnergyConservation,
    LiouvilleProton,
    LiouvilleElectron,
    Maxwell,
    VlasovProton,
    VlasovElectron,
    define_ind,
    define_velocity_field,
)
from model import NeuralNetwork, FNOModel

from modulus.sym.geometry.primitives_3d import Geometry, Sphere
from modulus.sym.utils.io.vtk import var_to_polyvtk


def make_geometry(cfg: ModulusConfig):
    nr_points: int = cfg.geometry.grid_resolution
    ball = Sphere(center=(0, 0, 0), radius=cfg.geometry.max_length)
    s = ball.sample_boundary(nr_points=nr_points)
    var_to_polyvtk(s, "general_geometry")
    return s


def define_nodes(model) -> List[Node]:
    maxwell = Maxwell()
    liouville_proton = LiouvilleProton()
    liouville_electron = LiouvilleElectron()
    vlasov_proton = VlasovProton()
    vlasov_electron = VlasovElectron()
    energy_conservation = EnergyConservation()

    nodes = (
        maxwell.nodes()
        + liouville_electron.nodes()
        + liouville_proton.nodes()
        + vlasov_electron.nodes()
        + vlasov_proton.nodes()
        + energy_conservation.nodes()
        + [model.make_nodes()]
    )

    return nodes


def define_pde_constraints(
    nodes: List[Node], geometry: Geometry, domain: Domain, cfg: ModulusConfig
) -> Domain:
    pde_residual = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geometry,
        outvar=dict(
            liouville_proton=0,
            liouville_electron=0,
            vlasov_proton=0,
            vlasov_electron=0,
            energy_conservation=0,
            gauss_elec=0,
            gauss_mag=0,
            faraday_x=0,
            faraday_y=0,
            faraday_z=0,
        ),
        batch_size=cfg.batch_size.pde,
        bounds={
            "x": cfg.bounds.r[0],
            "y": cfg.bounds.r[1],
            "z": cfg.bounds.r[2],
            "V_x": cfg.bounds.v[0],
            "V_y": cfg.bounds.v[1],
            "V_z": cfg.bounds.v[2],
            "t": cfg.bounds.t,
        },
    )
    domain.add_constraint(pde_residual, "pde_residual")
    return domain


def l1_boundary(nodes: List[Node], domain: Domain, cfg: ModulusConfig) -> Domain:
    train_dataset = DictGridDataset()  ## mirar como parametrizar
    l1_parameters = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.boundary1,
    )
    domain.add_constraint(l1_parameters, "l1_bc")
    return domain


def electron_boundary(
    nodes: List[Node], geometry: Geometry, domain: Domain, cfg: ModulusConfig
) -> Domain:
    x, y, z, t = define_ind()
    boltzmann_initial_condition = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geometry,
        outvar={},
        batch_size=cfg.batch_size.boundary,
        parameterization={x: 0, y: 0, z: 0, t: 0},
    )
    domain.add_constraint(boltzmann_initial_condition, "vlasov_electron_boundary")
    return domain


def proton_boundary(
    nodes: List[Node], geometry: Geometry, domain: Domain, cfg: ModulusConfig
) -> Domain:
    x, y, z, t = define_ind()
    boltzmann_initial_condition = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geometry,
        outvar={},
        batch_size=cfg.batch_size.boundary,
        parameterization={x: 0, y: 0, z: 0, t: 0},
    )
    domain.add_constraint(boltzmann_initial_condition, "vlasov_proton_boundary")
    return domain


def velocity_boundary(
    nodes: List[Node], geometry: Geometry, domain: Domain, cfg: ModulusConfig
) -> Domain:
    x, y, z, t = define_ind()
    input = dict(x=x, y=y, z=z, t=t)
    V_x, V_y, V_z = define_velocity_field(*input)
    velocity_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geometry,
        batch_size=cfg.batch_size.boundary,
        outvar={"f_e": 0, "f_p": 0},
        parameterization={
            V_x: Or(*cfg.bounds.v[0]),
            V_y: Or(*cfg.bounds.v[1]),
            V_z: Or(*cfg.bounds.v[2]),
        },
    )
    domain.add_constraint(velocity_bc, "velocity_bc")
    return domain


def define_boundary_conditions(
    nodes: List[Node], geometry: Geometry, domain: Domain, cfg: ModulusConfig
) -> Domain:
    boundary_conditions: List[Callable] = [
        electron_boundary,
        proton_boundary,
        velocity_boundary,
        l1_boundary,
    ]
    for func in boundary_conditions:
        domain: Domain = func(nodes, geometry, domain, cfg)
    return domain


@modulus.sym.main(config_path="conf", config_name="config")
def train_nn(cfg: ModulusConfig) -> None:
    model = NeuralNetwork(cfg)
    nodes: List[Node] = define_nodes(model)
    geometry: Geometry = make_geometry(cfg)
    domain: Domain = Domain()
    domain: Domain = define_pde_constraints(nodes, geometry, domain, cfg)
    domain: Domain = define_boundary_conditions(nodes, geometry, domain, cfg)
    slv = Solver(cfg, domain)
    slv.solve()


@modulus.sym.main(config_path="conf", config_name="config")
def train_fno(cfg: ModulusConfig) -> None:
    model = FNOModel(cfg)
    nodes: List[Node] = define_nodes(model)
    geometry: Geometry = make_geometry(cfg)
    domain: Domain = Domain()
    domain: Domain = define_pde_constraints(nodes, geometry, domain, cfg)
    domain: Domain = define_boundary_conditions(nodes, geometry, domain, cfg)
    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="VlasovMaxwell-PINN",
        description="Statistical mechanics informed Neural Network training for solar wind modeling",
    )
    parser.add_argument("-t", "--type", help="Define the architecture of the models")
    args = parser.parse_args()
    if args.type == "nn":
        train_nn()
    elif args.type == "fno":
        train_fno()
