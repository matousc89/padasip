import unittest
import sys
import numpy as np

sys.path.append('..')
import padasip as pa

class TestDetection(unittest.TestCase):

    def test_le_direct(self):
        """
        Learning entropy direct approach.
        """
        np.random.seed(100)
        n = 5
        N = 2000
        x = np.random.normal(0, 1, (N, n))
        d = np.sum(x, axis=1) + np.random.normal(0, 0.1, N)
        d[1000] += 2.
        f = pa.filters.FilterNLMS(n, mu=1., w=np.ones(n))
        y, e, w = f.run(d, x)
        le = pa.detection.learning_entropy(w, m=30, order=2)
        self.assertEqual(np.round(le.sum(), 3), 594.697)

    def test_le_multiscale(self):
        """
        Learning entropy multiscale approach.
        """
        np.random.seed(100)
        n = 5
        N = 2000
        x = np.random.normal(0, 1, (N, n))
        d = np.sum(x, axis=1) + np.random.normal(0, 0.1, N)
        d[1000] += 2.
        f = pa.filters.FilterNLMS(n, mu=1., w=np.ones(n))
        y, e, w = f.run(d, x)
        le = pa.detection.learning_entropy(w, m=30, order=2, alpha=[8., 9.])
        self.assertEqual(np.round(le.sum(), 3), 1.8)

    def test_elbnd(self):
        """
        ElBND
        """
        np.random.seed(100)
        n = 5
        N = 2000
        x = np.random.normal(0, 1, (N, n))
        d = np.sum(x, axis=1) + np.random.normal(0, 0.1, N)
        d[1000] += 2.
        f = pa.filters.FilterNLMS(n, mu=1., w=np.ones(n))
        y, e, w = f.run(d, x)
        elbnd = pa.detection.ELBND(w, e, function="max")
        self.assertEqual(np.round(elbnd.sum(), 3), 18.539)
