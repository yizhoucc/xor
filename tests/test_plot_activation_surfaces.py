import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.plot_activation_surfaces import load_learned_surface
from scripts.warmstart_free_init import InnerNetAct


class PlotActivationSurfacesTest(unittest.TestCase):
    def test_loads_real_checkpoint_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "inner.pth"
            torch.save(InnerNetAct(4).state_dict(), checkpoint)
            a, b, z = load_learned_surface(checkpoint, n=11, surface_range=2.0)
        self.assertEqual(a.shape, (11, 11))
        self.assertEqual(b.shape, (11, 11))
        self.assertEqual(z.shape, (11, 11))
        self.assertTrue(np.isfinite(z).all())


if __name__ == "__main__":
    unittest.main()
