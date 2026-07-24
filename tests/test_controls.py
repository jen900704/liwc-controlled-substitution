import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from liwc_controlled_substitution.controls import make_liwc_control


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.train = np.array([[1, 10, np.nan], [2, 20, 5], [3, 30, 7], [4, 40, 9]], float)
        self.test = np.array([[100, 1000, np.nan], [200, 2000, 8]], float)

    def test_pca_is_fold_local_and_width_preserving(self):
        train, test = make_liwc_control("pca", self.train, self.test, np.random.default_rng(1))
        self.assertEqual(train.shape, self.train.shape)
        self.assertEqual(test.shape, self.test.shape)
        self.assertTrue(np.isfinite(train).all())
        self.assertTrue(np.isfinite(test).all())
        self.assertTrue(np.allclose(train.mean(axis=0)[:2], 0, atol=1e-12))

    def test_shuffled_uses_complete_training_rows(self):
        train, test = make_liwc_control("shuffled", self.train, self.test, np.random.default_rng(2))
        filled = np.array([[1, 10, 7], [2, 20, 5], [3, 30, 7], [4, 40, 9]], float)
        self.assertEqual(sorted(map(tuple, train)), sorted(map(tuple, filled)))
        self.assertTrue(all(tuple(row) in set(map(tuple, filled)) for row in test))

    def test_random_samples_each_training_marginal(self):
        train, test = make_liwc_control("random", self.train, self.test, np.random.default_rng(3))
        combined = np.vstack([train, test])
        allowed = [{1, 2, 3, 4}, {10, 20, 30, 40}, {5, 7, 9}]
        for column in range(3):
            self.assertTrue(set(combined[:, column]).issubset(allowed[column]))


if __name__ == "__main__":
    unittest.main()
