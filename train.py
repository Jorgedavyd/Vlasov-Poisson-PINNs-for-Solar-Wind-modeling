from modulus.sym.domain.constraint.discrete import DictGridDataset
from modulus.sym.hydra import ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from numpy._typing import NDArray
from scipy.interpolate import CubicSpline
from sympy import Or
from argparse import ArgumentParser
from typing import List, Callable, Tuple
import modulus.sym
from modulus.sym import Node, Key
from modulus.sym.models.fno.fno import FNO
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
from model import NeuralNetwork
import numpy as np
from modulus.sym.geometry.primitives_3d import Geometry, Sphere
from modulus.sym.utils.io.vtk import var_to_polyvtk
import pandas as pd


def make_geometry(cfg: ModulusConfig):
    nr_points: int = cfg.custom.geometry.grid_resolution
    ball = Sphere(center=(0, 0, 0), radius=cfg.custom.geometry.r[-1])
    s = ball.sample_boundary(nr_points=nr_points)
    var_to_polyvtk(s, "general_geometry")
    return s


def define_pde(*args) -> List[Node]:
    nodes = []
    for arg in args:
        nodes.append(arg.make_nodes())

    maxwell = Maxwell()

    liouville_proton = Liouville()
    vlasov_proton = Vlasov()
    continuity_proton = Continuity()

    liouville_electron = Liouville()
    vlasov_electron = Vlasov()
    continuity_electron = Continuity()

    nodes = (
        maxwell.nodes()
        + liouville_electron.nodes()
        + liouville_proton.nodes()
        + continuity_proton.nodes()
        + vlasov_electron.nodes()
        + vlasov_proton.nodes()
        + continuity_electron.nodes()
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
            "x": cfg.custom.bounds.r[0],
            "y": cfg.custom.bounds.r[1],
            "z": cfg.custom.bounds.r[2],
            "V_x": cfg.custom.bounds.v[0],
            "V_y": cfg.custom.bounds.v[1],
            "V_z": cfg.custom.bounds.v[2],
            "t": cfg.custom.bounds.t,
        },
    )
    domain.add_constraint(pde_residual, "pde_residual")
    return domain


def get_interpolators(variables: List[str], path: str) -> Tuple[CubicSpline]:
    interpolators: List[CubicSpline] = list()
    data: pd.DataFrame = pd.read_csv(path)
    time: NDArray = data.index.values.astype(np.float32)
    for variable in variables:
        interpolators.append(CubicSpline(time, data[variable].values))
    return interpolators


def l1_boundary(nodes: List[Node], domain: Domain, cfg: ModulusConfig) -> Domain:
    path: str = "/data/Vlasov/curated.csv"
    variables: List[str] = [
        "E_x",
        "E_y",
        "E_z",
        "B_x",
        "B_y",
        "B_z",
        "V_x",
        "V_y",
        "V_z",
    ]
    time_grid: NDArray = np.arange(0, 1, cfg.custom.geometry.grid_resolution)
    (
        E_x_interpolator,
        E_y_interpolator,
        E_z_interpolator,
        B_x_interpolator,
        B_y_interpolator,
        B_z_interpolator,
        V_x_interpolator,
        V_y_interpolator,
        V_z_interpolator,
    ) = get_interpolators(variables, path)
    train_dataset = DictGridDataset(
        invar=dict(
            x=np.zeros_like(time_grid),
            y=np.zeros_like(time_grid),
            z=np.zeros_like(time_grid),
            t=time_grid,
        ),
        outvar=dict(
            E_x=E_x_interpolator(time_grid),
            E_y=E_y_interpolator(time_grid),
            E_z=E_z_interpolator(time_grid),
            B_x=B_x_interpolator(time_grid),
            B_y=B_y_interpolator(time_grid),
            B_z=B_z_interpolator(time_grid),
            V_x=V_x_interpolator(time_grid),
            V_y=V_y_interpolator(time_grid),
            V_z=V_z_interpolator(time_grid),
        ),
    )
    l1_parameters = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.boundary,
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
            V_x: Or(*cfg.custom.bounds.v[0]),
            V_y: Or(*cfg.custom.bounds.v[1]),
            V_z: Or(*cfg.custom.bounds.v[2]),
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
def run(cfg: ModulusConfig) -> None:
    f_p, f_e, f_E, f_B = ...
    nodes: List[Node] = define_pde(f_p, f_e, f_E, f_B)
    geometry: Geometry = make_geometry(cfg)
    domain: Domain = Domain()
    domain: Domain = define_pde_constraints(nodes, geometry, domain, cfg)
    domain: Domain = define_boundary_conditions(nodes, geometry, domain, cfg)
    slv = Solver(cfg, domain)
    slv.solve()


@modulus.sym.main(config_path="conf", config_name="config")
def train_fno(cfg: ModulusConfig) -> None:
    f_p = FNOModel(cfg.arch.f_p.fno)
    model = FNOModel(cfg)
    nodes: List[Node] = define_pde(model)
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
