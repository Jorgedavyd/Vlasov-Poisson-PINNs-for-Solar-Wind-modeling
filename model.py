from lightorch.nn import DeepNeuralNetwork
from modulus.sym.models.arch import Arch
from modulus.sym.models.fno.fno import FNO
from modulus.sym.hydra import ModulusConfig
from modulus.sym.key import Key
from typing import Dict
from torch import nn
import torch
from torch import Tensor


class NeuralNetwork(Arch):
    def __init__(self, cfg: ModulusConfig):
        super().__init__(
            input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
            output_keys=[
                Key("E_x"),
                Key("E_y"),
                Key("E_z"),
                Key("B_x"),
                Key("B_y"),
                Key("B_z"),
                Key("V_x"),
                Key("V_y"),
                Key("V_z"),
                Key("f_e"),
                Key("f_p"),
            ],
        )

        self.f_e = DeepNeuralNetwork(
            7,
            cfg.arch.neural_network.f_e.hidden_layers,
            [getattr(nn, f) for f in cfg.arch.neural_network.f_e.activations],
        )

        self.f_p = DeepNeuralNetwork(
            7,
            cfg.arch.neural_network.f_p.hidden_layers,
            [getattr(nn, f) for f in cfg.arch.neural_network.f_p.activations],
        )

        self.f_v = DeepNeuralNetwork(
            4,
            cfg.arch.neural_network.f_v.hidden_layers,
            [getattr(nn, f) for f in cfg.arch.neural_network.f_v.activations],
        )

        self.f_E = DeepNeuralNetwork(
            4,
            cfg.arch.neural_network.f_E.hidden_layers,
            [getattr(nn, f) for f in cfg.arch.neural_network.f_E.activations],
        )

        self.f_B = DeepNeuralNetwork(
            4,
            cfg.arch.neural_network.f_B.hidden_layers,
            [getattr(nn, f) for f in cfg.arch.neural_network.f_B.activations],
        )

    def forward(self, dict_tensor: Dict[str, Tensor]):
        input: Tensor = self.concat_input(
            {k: dict_tensor[k] for k in list(self.input_keys_.keys())},
            list(self.input_keys_.keys()),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )

        E: Tensor = self.f_E(input)
        B: Tensor = self.f_B(input)
        V: Tensor = self.f_v(input)
        f_input: Tensor = torch.cat([input, V], dim=-1)
        f_e_output: Tensor = self.f_e(f_input, dim=-1)
        f_p_output: Tensor = self.f_p(f_input, dim=-1)

        return {
            "E_x": E[:, :, 0],
            "E_y": E[:, :, 1],
            "E_z": E[:, :, 2],
            "B_x": B[:, :, 0],
            "B_y": B[:, :, 1],
            "B_z": B[:, :, 2],
            "V_x": V[:, :, 0],
            "V_y": V[:, :, 1],
            "V_z": V[:, :, 2],
            "f_e": f_e_output,
            "f_p": f_p_output,
        }


class FNOModel(Arch):
    def __init__(self, cfg: ModulusConfig) -> None:
        super().__init__(
            input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
            output_keys=[
                Key("E_x"),
                Key("E_y"),
                Key("E_z"),
                Key("B_x"),
                Key("B_y"),
                Key("B_z"),
                Key("V_x"),
                Key("V_y"),
                Key("V_z"),
                Key("f_e"),
                Key("f_p"),
            ],
        )

        self.f_e = FNO(
            input_keys=[
                Key("x"),
                Key("y"),
                Key("z"),
                Key("V_x"),
                Key("V_y"),
                Key("V_z"),
                Key("t"),
            ],
            **cfg.arch.fno.f_e
        )

        self.f_p = FNO(
            input_keys=[
                Key("x"),
                Key("y"),
                Key("z"),
                Key("V_x"),
                Key("V_y"),
                Key("V_z"),
                Key("t"),
            ],
            **cfg.arch.fno.f_p
        )

        self.f_v = FNO(input_keys=self.input_keys, **cfg.arch.fno.f_v)

        self.f_E = FNO(input_keys=self.input_keys, **cfg.arch.fno.f_E)

        self.f_B = FNO(input_keys=self.input_keys, **cfg.arch.fno.f_B)

    def forward(self, input_var: Dict[str, Tensor]) -> Dict[str, Tensor]:
        E_output = self.f_E(input_var)
        B_output = self.f_B(input_var)
        V_output = self.f_v(input_var)

        second_input: Dict[str, Tensor] = {}
        second_input.update(input_var)
        second_input.update(V_output)
        second_input.update(E_output)
        second_input.update(B_output)

        f_p_output = self.f_p(second_input)
        f_e_output = self.f_e(second_input)

        output: Dict[str, Tensor] = {}
        output.update(f_e_output)
        output.update(f_p_output)
        output.update(V_output)
        output.update(E_output)
        output.update(B_output)

        return output
