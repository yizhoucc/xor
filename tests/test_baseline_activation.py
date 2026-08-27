import unittest
from types import SimpleNamespace

import torch.nn as nn

from model.baseline import BaselineCNN, BaselineMLP


def _config(model):
    return SimpleNamespace(
        model=SimpleNamespace(**model),
        dataset=SimpleNamespace(name="cifar10"),
    )


class BaselineActivationTest(unittest.TestCase):
    def test_cnn_supports_prelu(self):
        config = _config({
            "input_channel": 3, "out_channel": [4, 8],
            "kernel_size": [3, 1], "zero_pad": [0, 0], "stride": [1, 1],
            "dropout": 0.0, "use_layernorm": True, "activation": "prelu",
            "num_classes": 10, "loss": "CrossEntropy",
        })
        network = BaselineCNN(config)
        self.assertEqual(sum(isinstance(layer, nn.PReLU) for layer in network.features), 2)

    def test_mlp_supports_swish_alias(self):
        config = _config({
            "input_dim": 12, "out_hidden_dim": [8, 8], "dropout": 0.0,
            "activation": "swish", "num_classes": 3, "loss": "CrossEntropy",
        })
        network = BaselineMLP(config)
        self.assertEqual(sum(isinstance(layer, nn.SiLU) for layer in network.hidden), 2)


if __name__ == "__main__":
    unittest.main()
