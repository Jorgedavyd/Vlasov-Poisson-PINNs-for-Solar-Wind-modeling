from modulus.sym.domain.constraint.discrete import DictGridDataset
from modulus.sym.hydra import ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from numpy._typing import NDArray
from sympy import Symbol
from typing import List, Callable
import modulus.sym
from modulus.sym import Node, Key
from modulus.sym.hydra import instantiate_arch
from mudulus.sym.loss.loss import DecayedPointwiseLossNorm
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    SupervisedGridConstraint,
)
from equations import (
    Vlasov,
    Continuity,
    Maxwell,
    Velocity,
    Density,
    define_ind,
    define_velocity,
)
import numpy as np
from modulus.sym.geometry.primitives_3d import Geometry, Sphere
from modulus.sym.utils.io.vtk import var_to_polyvtk
import pandas as pd


def make_geometry(cfg: ModulusConfig):
    nr_points: int = cfg.custom.geometry.nr_points
    ball = Sphere(center=(0, 0, 0), radius=cfg.custom.bounds.r[-1])
    s = ball.sample_boundary(nr_points=nr_points)
    var_to_polyvtk(s, "general_geometry")
    return s


def define_nodes(
    geometry: Geometry, nr_points: int, montecarlo_points: int, *args
) -> List[Node]:
    maxwell = Maxwell()
    vlasov_proton = Vlasov("p")
    continuity_proton = Continuity("p")
    vlasov_electron = Vlasov("e")
    continuity_electron = Continuity("e")

    nodes = (
        [
            Node(
                ["f_e", "t"],
                ["rho_e"],
                Density("e", geometry, nr_points, montecarlo_points),
            ),
            Node(
                ["f_p", "t"],
                ["rho_p"],
                Density("p", geometry, nr_points, montecarlo_points),
            ),
            Node(
                ["f_e", "t"],
                ["v_xe", "v_ye", "v_ze"],
                Velocity("e", geometry, nr_points, montecarlo_points),
            ),
            Node(
                ["f_p", "t"],
                ["v_xp", "v_yp", "v_zp"],
                Velocity("p", geometry, nr_points, montecarlo_points),
            ),
        ]
        + maxwell.make_nodes()
        + continuity_proton.make_nodes()
        + vlasov_electron.make_nodes()
        + vlasov_proton.make_nodes()
        + continuity_electron.make_nodes()
    )

    for arg in args:
        nodes.append(arg.make_nodes())

    return nodes


def define_pde_constraints(
    nodes: List[Node], geometry: Geometry, domain: Domain, cfg: ModulusConfig
) -> Domain:
    x, y, z, t = define_ind()
    pde_residual = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geometry,
        outvar=dict(
            vlasov_proton=0,
            vlasov_electron=0,
            continuity_proton=0,
            continuity_electron=0,
            gauss_elec=0,
            gauss_mag=0,
            faraday_x=0,
            faraday_y=0,
            faraday_z=0,
            ampere_x=0,
            ampere_y=0,
            ampere_z=0,
        ),
        batch_size=cfg.batch_size.pde,
        bounds={
            x: tuple(cfg.custom.bounds.r[0]),
            y: tuple(cfg.custom.bounds.r[1]),
            z: tuple(cfg.custom.bounds.r[2]),
        },
        parameterization={
            t: tuple(cfg.custom.bounds.t),
        },
    )
    domain.add_constraint(pde_residual, "pde_residual")
    return domain


def l1_boundary(nodes: List[Node], domain: Domain, cfg: ModulusConfig) -> Domain:
    path: str = "/data/Vlasov/curated.csv"
    variables: List[str] = [
        "time",
        "E_x",
        "E_y",
        "E_z",
        "B_x",
        "B_y",
        "B_z",
        "rho_e",
        "rho_p",
        "v_xp",
        "v_yp",
        "v_zp",
        "v_xe",
        "v_ye",
        "v_ze",
        "f_p",
        "f_e",
    ]
    df: pd.DataFrame = pd.read_csv(path)
    time: NDArray = df.values

    train_dataset = DictGridDataset(
        invar=dict(
            x=np.zeros_like(time),
            y=np.zeros_like(time),
            z=np.zeros_like(time),
            t=time,
        ),
        outvar={k: df[k].values for k in variables},
    )

    l1_parameters = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.boundary,
        loss=DecayedPointwiseLossNorm(),
    )

    domain.add_constraint(l1_parameters, "l1_bc")
    return domain


def velocity_boundary(
    nodes: List[Node], geometry: Geometry, domain: Domain, cfg: ModulusConfig
) -> Domain:
    x, y, z, t = define_ind()
    input = dict(x=x, y=y, z=z, t=t)
    vxe, vye, vze = define_velocity(input, "e")
    vxp, vyp, vzp = define_velocity(input, "p")
    velocity_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geometry,
        batch_size=cfg.batch_size.boundary,
        outvar={"f_e": 0, "f_p": 0},
        loss=DecayedPointwiseLossNorm(),
        parameterization={
            vxe: 0,
            vye: 0,
            vze: 0,
            vxp: 0,
            vyp: 0,
            vzp: 0,
        },
    )
    domain.add_constraint(velocity_bc, "velocity_bc")
    return domain


def define_boundary_conditions(
    nodes: List[Node], geometry: Geometry, domain: Domain, cfg: ModulusConfig
) -> Domain:
    boundary_conditions: List[Callable] = [
        velocity_boundary,
        l1_boundary,
    ]
    for idx, func in enumerate(boundary_conditions):
        domain.add_constraint(
            func(nodes, geometry, domain, cfg),
            str(idx),
        )

    return domain


def initial_conditions(
    nodes: List[Node], geometry: Geometry, domain: Domain, cfg: ModulusConfig
) -> Domain:
    t = Symbol("t")
    initial = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geometry,
        outvar=dict(
            f_p=0,
            f_e=0,
            E_x=0,
            E_y=0,
            E_z=0,
            B_x=0,
            B_y=0,
            B_z=0,
            v_xp=0,
            v_yp=0,
            v_zp=0,
            v_xe=0,
            v_ye=0,
            v_ze=0,
        ),
        batch_size=cfg.batch_size.boundary,
        loss=DecayedPointwiseLossNorm(),
        parameterization={t: 0},
    )
    domain.add_constraint(initial, "initial_conditions")
    return domain


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    f_p, f_e, E, B = (
        instantiate_arch(
            invar=[Key("x"), Key("y"), Key("z"), Key("v_xp"), Key("v_yp"), ("v_zp")],
            outvar=[Key("f_p")],
            cfg=cfg.arch.f_p,
        ),
        instantiate_arch(
            invar=[Key("x"), Key("y"), Key("z"), Key("v_xe"), Key("v_ye"), ("v_ze")],
            outvar=[Key("f_e")],
            cfg=cfg.arch.f_e,
        ),
        instantiate_arch(
            invar=[Key("x"), Key("y"), Key("z"), Key("t")],
            outvar=[Key("E_x"), Key("E_y"), Key("E_z")],
            cfg=cfg.arch.E,
        ),
        instantiate_arch(
            invar=[Key("x"), Key("y"), Key("z"), Key("t")],
            outvar=[Key("B_x"), Key("B_y"), Key("B_z")],
            cfg=cfg.arch.B,
        ),
    )

    geometry: Geometry = make_geometry(cfg)
    nodes: List[Node] = define_nodes(
        geometry,
        cfg.custom.geometry.nr_points,
        cfg.custom.geometry.montecarlo_points,
        f_p,
        f_e,
        E,
        B,
    )

    domain: Domain = Domain()
    domain: Domain = initial_conditions(nodes, geometry, domain, cfg)
    domain: Domain = define_boundary_conditions(nodes, geometry, domain, cfg)
    domain: Domain = define_pde_constraints(nodes, geometry, domain, cfg)

    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()
