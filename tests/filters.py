import unittest
import sys
import numpy as np


sys.path.append('..')
import padasip as pa


class TestFilters(unittest.TestCase):

    def test_base_filter_adapt(self):
        filt = pa.filters.FilterLMS(3, mu=1., w="zeros")
        x = np.array([2, 4, 3])
        filt.adapt(1, x)
        self.assertAlmostEqual(filt.w.sum(), 9.0)

    def test_filter_gngd(self):
        """
        Test of GNGD filter output.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterGNGD(n=4, mu=0.9, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), 16.622071160225627)

    def test_filter_vslms_mathews(self):
        """
        Test of VLSMS with Mathews adaptation filter output.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterVSLMS_Mathews(n=4, mu=0.1, ro=0.001, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), 18.46303593650432)

    def test_filter_vslms_benveniste(self):
        """
        Test of VLSMS with Benveniste adaptation filter output.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterVSLMS_Benveniste(n=4, mu=0.1, ro=0.0002, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), 18.048916937718058)

    def test_filter_vslms_ang(self):
        """
        Test of VLSMS with Ang adaptation filter output.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterVSLMS_Ang(n=4, mu=0.1, ro=0.0002, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), 18.341053442007972)

    def test_filter_ap(self):
        """
        Test of AP filter output.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterAP(n=4, order=5, mu=0.5, ifc=0.001, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), 15.105550229065491)

    def test_filter_lms(self):
        """
        Test of LMS filter output.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterLMS(n=4, mu=0.1, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), 18.199308184867885)

    def test_filter_nlms(self):
        """
        Test of NLMS filter.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterNLMS(n=4, mu=0.5, eps=0.01, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), 14.246570369497373)

    def test_filter_ocnlms(self):
        """
        Test of OCNLMS filter.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterOCNLMS(n=4, mu=1., mem=100, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), 13.775034155354426)

    def test_filter_Llncosh(self):
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterLlncosh(n=4, mu=1., lambd=3, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), 18.74164638623726)

    def test_filter_rls(self):
        """
        Test of RLS filter.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterRLS(n=4, mu=0.9, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), 16.80842884997325)

    def test_filter_LMF(self):
        """
        Test of LMF filter.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 1*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterLMF(n=4, mu=0.01, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), 16.611322392961064)

    def test_filter_NLMF(self):
        """
        Test of NLMF filter.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 1*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterNLMF(n=4, mu=0.1, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), 13.989262305958494)

    def test_filter_SSLMS(self):
        """
        Test of SSLMS filter.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 1*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterSSLMS(n=4, mu=0.1, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), 12.579327704869938)

    def test_filter_NSSLMS(self):
        """
        Test of NSSLMS filter.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 1*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterNSSLMS(n=4, mu=0.3, eps=0.001, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), -23.49342599164458)

    def test_filter_GMCC(self):
        """
        Test of GMCC filter.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 1*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterGMCC(n=4, mu=0.3, lambd=0.03, alpha=2, w="random")
        y, e, w = f.run(d, x)
        self.assertAlmostEqual(y.sum(), 7.002285017142926)
