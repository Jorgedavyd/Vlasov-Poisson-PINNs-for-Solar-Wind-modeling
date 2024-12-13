from lightorch.nn import DeepNeuralNetwork
from modulus.sym.models.arch import Arch
from modulus.hydra import ModulusConfig
from modulus.key import Key
from typing import Dict
from torch import nn
import torch


class NeuralNetwork(Arch):
    def __init__(self, cfg: ModulusConfig):
        if cfg.dimensions == 1:
            self.input_keys_ = dict(
                x=Key("x"),
            )
        elif cfg.dimensions == 2:
            self.input_keys_ = dict(
                x=Key("x"),
                y=Key("y"),
            )
        elif cfg.dimensions == 3:
            self.input_keys_ = dict(
                x=Key("x"),
                y=Key("y"),
                z=Key("z"),
            )
        self.input_keys_.update(dict(t=Key("t")))

        super().__init__(
            input_keys=self.input_keys_,
            output_keys=dict(
                E=Key("E"),
                B=Key("B"),
                v=Key("v"),
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

    def forward(self, dict_tensor: Dict[str, torch.Tensor]):
        input = self.concat_input(
            {k: dict_tensor[k] for k in list(self.input_keys_.keys())},
            list(self.input_keys_.keys()),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )

        out = self.model(input)

        return self.split_output(out, self.output_key_dict, dim=-1)
