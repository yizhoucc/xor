import unittest

from scripts.plot_parameter_efficiency import count_parameters


class PlotParameterEfficiencyTest(unittest.TestCase):
    def test_expected_cross_width_parameter_saving(self):
        innernet = count_parameters("config/experiments/mlp_cifar_scale_2arg_w128.yaml")
        relu = count_parameters("config/experiments/mlp_cifar_scale_relu_w256.yaml")
        self.assertEqual(innernet, 415051)
        self.assertEqual(relu, 920842)
        self.assertAlmostEqual(100 * (1 - innernet / relu), 54.93, places=2)


if __name__ == "__main__":
    unittest.main()
