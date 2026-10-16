import argparse
import unittest

import numpy as np

from adaptive_basin_fvm import (
    AdaptiveConfig,
    AdaptiveMesh,
    Cell,
    adaptive_step,
    balance_cells,
    face_pairs,
)
from run_adaptive_grid import parse_case


class AdaptiveMeshTests(unittest.TestCase):
    def test_coarse_fine_face_lengths_are_conservative(self):
        cells = [Cell(0, 4, 0, 4, 0)] + [Cell(4, 5, j, j + 1, 2) for j in range(4)]
        pairs = face_pairs(cells, 1.0, 1.0, (0.0, 8.0, 0.0, 4.0))
        touching = pairs["p"] == 0
        self.assertAlmostEqual(float(pairs["length"][touching].sum()), 4.0)

    def test_balance_limits_neighbour_level_jump(self):
        cells = [Cell(0, 4, 0, 4, 0)] + [Cell(4, 5, j, j + 1, 2) for j in range(4)]
        balanced = balance_cells(cells, 2)
        for a, b, *_ in __import__("adaptive_basin_fvm")._integer_face_pairs(balanced):
            self.assertLessEqual(abs(balanced[a].level - balanced[b].level), 1)

    def test_constant_temperature_reports_picard_convergence(self):
        config = AdaptiveConfig(base_nx=2, base_nz=2, max_level=0)
        cells = [Cell(i, i + 1, j, j + 1, 0) for j in range(2) for i in range(2)]
        mesh = AdaptiveMesh(cells, (0.0, 2.0, 0.0, 2.0), config)
        temperature = np.full(4, 20.0)
        grid = {
            "active": np.ones(4, dtype=bool),
            "lith": np.zeros(4, dtype=int),
            "phi": np.zeros(4),
        }
        state = {
            "surface": np.zeros(4),
            "base": np.full(4, 2.0),
            "tsurf": np.full(4, 20.0),
            "qbasal": np.zeros(4),
        }
        properties = {0: {"lambda20": 2.5, "c20": 900.0, "rho_s": 2500.0, "As": 0.0}}

        result, diagnostics = adaptive_step(
            temperature, mesh, state, grid, 1.0e6, properties, config
        )

        np.testing.assert_allclose(result, temperature)
        self.assertTrue(diagnostics["nonlinear_converged"])
        self.assertEqual(diagnostics["picard_iterations"], 1)

    def test_invalid_solver_controls_are_rejected(self):
        with self.assertRaises(ValueError):
            AdaptiveConfig(picard_iterations=0).validate()
        with self.assertRaises(ValueError):
            AdaptiveConfig(picard_tolerance_c=0.0).validate()
        with self.assertRaises(ValueError):
            AdaptiveConfig(linear_tolerance=0.0).validate()

    def test_adaptive_grid_case_parser(self):
        self.assertEqual(parse_case("8, 16, 3"), (8, 16, 3))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_case("8,16,6")


if __name__ == "__main__":
    unittest.main()
