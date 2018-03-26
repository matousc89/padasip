import unittest
import sys
import numpy as np


sys.path.append('..')
import padasip as pa


class TestFilters(unittest.TestCase):

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

    def test_filter_ap(self):
        """
        Test of AP filter output.
        """
        np.random.seed(100)
        N = 100
        x = np.random.normal(0, 1, (N, 4))
        v = np.random.normal(0, 0.1, N)
        d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v
        f = pa.filters.FilterAP(n=4, order=5, mu=0.5, eps=0.001, w="random")
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
        f = pa.filters.FilterNLMS(n=4, mu=1., eps=1., w="random")
        y, e, w = f.run(d, x)    
        self.assertAlmostEqual(y.sum(), 17.075790883173546)

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
        self.assertAlmostEqual(y.sum(), 12.466854176928789)

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
        self.assertAlmostEqual(y.sum(), 12.236638660551113)

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
        f = pa.filters.FilterSSLMS(n=4, mu=0.3, w="random")
        y, e, w = f.run(d, x)   
        self.assertAlmostEqual(y.sum(), 21.982245163799284)

    def test_filters_length(self):
        """
        Test if filters does not accept incorrect length of filter
        Done for: LMS, NLMS, AP, GNGD, RLS
        """
        filters = [
                pa.filters.FilterLMS,
                pa.filters.FilterNLMS, 
                pa.filters.FilterAP,
                pa.filters.FilterRLS,
                pa.filters.FilterGNGD,
                pa.filters.FilterLMF,
                pa.filters.FilterNLMF,
                pa.filters.FilterSSLMS,
                pa.filters.FilterNSSLMS,
                ]
        # test if works
        for item in filters:
            f = item(n=4)
        # test errors 
        for item in filters:        
            with self.assertRaises(ValueError):
                f = item(n=4.) # n is float
            with self.assertRaises(ValueError):
                f = item(n=" ") # n is string
            with self.assertRaises(ValueError):
                f = item(n=-4) # n is negative
            with self.assertRaises(ValueError):
                f = item(n=[1,2]) # n is array

    def test_filters_learning_rate(self):
        """
        Test if filters does not accept incorrect mu.
        Done for: LMS, NLMS, AP, RLS, GNGD
        """
        filters = [
                pa.filters.FilterLMS,
                pa.filters.FilterNLMS, 
                pa.filters.FilterAP,
                pa.filters.FilterRLS,
                pa.filters.FilterGNGD,
                pa.filters.FilterLMF,
                pa.filters.FilterNLMF,
                pa.filters.FilterSSLMS,
                pa.filters.FilterNSSLMS,
                ]
        # test if works
        for item in filters:
            f = item(n=4, mu=0.1)
        # test errors 
        for item in filters:        
            with self.assertRaises(ValueError):
                f = item(n=4, mu=" ") # mu is str
            with self.assertRaises(ValueError):
                f = item(n=4, mu=[1,2]) # mu is array
            with self.assertRaises(ValueError):
                f = item(n=4, mu=-1) # mu is negative

    def test_filters_epsilon(self):
        """
        Test if filters does not accept incorrect epsilon.
        NLMS, AP, RLS, GNGD
        """
        filters = [
                pa.filters.FilterNLMS, 
                pa.filters.FilterAP,
                pa.filters.FilterRLS,
                pa.filters.FilterGNGD,
                pa.filters.FilterNLMF,
                pa.filters.FilterNSSLMS,
                ]  
        # test if works
        for item in filters:
            f = item(n=4, eps=0.1)
        # test errors    
        for item in filters:        
            with self.assertRaises(ValueError):
                f = item(n=4, eps=" ") # eps is str
            with self.assertRaises(ValueError):
                f = item(n=4, eps=[1,2]) # eps is array
            with self.assertRaises(ValueError):
                f = item(n=4, eps=-1) # eps is negative
                
    def test_filter_helpers(self):
        """
        Test AdaptiveFilter helper, if it is require the n
        """
        with self.assertRaises(ValueError):
            f = pa.filters.AdaptiveFilter() # n is not provided    
    
    
    
    


