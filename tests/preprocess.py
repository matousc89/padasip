import unittest
import sys
import numpy as np


sys.path.append('..')
import padasip as pa

class TestPreprocess(unittest.TestCase):

    def test_standardize(self):
        """
        Test standardization.
        """
        u = range(1000)
        x = pa.standardize(u)
        self.assertEqual(x.std(), 1.0)
        self.assertEqual(np.round(x.mean(), 3), -0.0)
        
    def test_standardize_back(self):
        """
        Test de-standardization.
        """
        x = range(1000)
        u = pa.standardize_back(x, 2, 10)
        self.assertEqual(np.round(u.std(), 3), 2886.75)
        self.assertEqual(np.round(u.mean(), 3), 4997.0)
        
    def test_input_from_history(self):
        """
        Test input from history function.
        """
        u = range(1000)
        x = pa.input_from_history(u, 4)       
        self.assertEqual(x.shape, (997, 4))
        self.assertEqual(np.round(np.array(u).mean(), 5), np.round(x.mean(), 5))
        

