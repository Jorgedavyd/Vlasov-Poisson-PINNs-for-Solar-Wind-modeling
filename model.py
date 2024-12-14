from lightorch.nn import DeepNeuralNetwork
from modulus.sym.models.arch import Arch
from modulus.hydra import ModulusConfig
from modulus.key import Key
from typing import Dict
from torch import nn
import torch
from torch import Tensor

class NeuralNetwork(Arch):
    def __init__(self, cfg: ModulusConfig):
        super().__init__(
            input_keys = dict(
                x=Key("x"),
                y=Key("y"),
                z=Key("z"),
                t=Key("t")
            ),
            output_keys=dict(
                E_x=Key("E_x"),
                E_y=Key("E_y"),
                E_z=Key("E_z"),
                B_x=Key("B_x"),
                B_y=Key("B_y"),
                B_z=Key("B_z"),
                V_x=Key("V_x"),
                V_y=Key("V_y"),
                V_z=Key("V_z"),
                f_e=Key("f_e"),
                f_p=Key("f_p"),
            ),
        )

        self.f_e = DeepNeuralNetwork(
            cfg.dimensions * 2 + 1,
            cfg.neural_network.f_e.hidden_layers,
            [getattr(nn, f) for f in cfg.neural_network.f_e.activations],
        )

        self.f_p = DeepNeuralNetwork(
            cfg.dimensions * 2 + 1,
            cfg.neural_network.f_p.hidden_layers,
            [getattr(nn, f) for f in cfg.neural_network.f_p.activations],
        )

        self.f_v = DeepNeuralNetwork(
            cfg.dimensions + 1,
            cfg.neural_network.f_v.hidden_layers,
            [getattr(nn, f) for f in cfg.neural_network.f_v.activations],
        )

        self.f_E = DeepNeuralNetwork(
            cfg.dimensions + 1,
            cfg.neural_network.f_E.hidden_layers,
            [getattr(nn, f) for f in cfg.neural_network.f_E.activations],
        )

        self.f_B = DeepNeuralNetwork(
            cfg.dimensions + 1,
            cfg.neural_network.f_B.hidden_layers,
            [getattr(nn, f) for f in cfg.neural_network.f_B.activations],
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
        f_input: Tensor = torch.cat([input, V], dim = -1)
        f_e_output: Tensor = self.f_e(f_input, dim = -1)
        f_p_output: Tensor = self.f_p(f_input, dim = -1)

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
