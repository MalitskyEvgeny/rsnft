import unittest

import numpy as np

from basin_fvm import harmonic, sekiguchi_conductivity, waples_heat_capacity


class CoreFormulaTests(unittest.TestCase):
    def test_harmonic_mean(self):
        self.assertAlmostEqual(harmonic(2.0, 4.0), 8.0 / 3.0)

    def test_waples_reference_value_at_20c(self):
        value = waples_heat_capacity(np.array([293.15]), np.array([900.0]))[0]
        self.assertAlmostEqual(value, 897.9079752, places=5)

    def test_sekiguchi_is_lambda20_at_20c_to_roundoff(self):
        value = sekiguchi_conductivity(np.array([293.15]), np.array([3.0]))[0]
        self.assertAlmostEqual(value, 3.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()
