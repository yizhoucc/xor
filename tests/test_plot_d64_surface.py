import tempfile
import unittest
from pathlib import Path

import torch

from scripts.plot_d64_surface import build_surfaces
from scripts.warmstart_free_init import InnerNetAct


class PlotD64SurfaceTest(unittest.TestCase):
    def test_build_surfaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "inner.pth"
            torch.save(InnerNetAct(4).state_dict(), checkpoint)
            surfaces = build_surfaces(checkpoint, n=12, surface_range=2.0)
        self.assertEqual(surfaces["learned"].shape, (12, 12))
        self.assertEqual(surfaces["fitted"].shape, (12, 12))
        self.assertIn("r2", surfaces["fit"])


if __name__ == "__main__":
    unittest.main()
