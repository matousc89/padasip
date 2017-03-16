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
        self.assertEqual(np.round(y.sum(), 4), 16.6221)

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
        self.assertEqual(np.round(y.sum(), 4), 15.1056)

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
        self.assertEqual(np.round(y.sum(), 4), 18.1993)

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
        self.assertEqual(np.round(y.sum(), 4), 17.0758)

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
        self.assertEqual(np.round(y.sum(), 4), 12.4669)

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
    
    
    
    


