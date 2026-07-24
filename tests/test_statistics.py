import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from liwc_controlled_substitution.statistics import benjamini_hochberg, exact_sign_flip_test


class StatisticsTests(unittest.TestCase):
    def test_exact_sign_flip(self):
        self.assertEqual(exact_sign_flip_test([1, 1, 1]), 0.25)
        self.assertEqual(exact_sign_flip_test([1, -1]), 1.0)
        self.assertTrue(np.isnan(exact_sign_flip_test([np.nan])))

    def test_bh_known_values_and_nan(self):
        result = benjamini_hochberg([0.01, 0.04, 0.03, 0.002, np.nan])
        expected = np.array([0.02, 0.04, 0.04, 0.008, np.nan])
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_bh_rejects_invalid_p(self):
        with self.assertRaises(ValueError):
            benjamini_hochberg([1.1])


if __name__ == "__main__":
    unittest.main()
