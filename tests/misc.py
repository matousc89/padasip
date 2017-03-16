import unittest
import sys
import numpy as np

sys.path.append('..')
import padasip as pa

class TestErrorEval(unittest.TestCase):

    def test_MSE(self):
        """
        Test MSE
        """
        # two inputs
        x1 = np.array([1, 2, 3, 4, 5])
        x2 = np.array([5, 4, 3, 2, 1])
        mse = pa.misc.MSE(x1, x2)
        self.assertEqual(mse, 8.0)        
        # one input
        e = x1 - x2
        mse = pa.misc.MSE(e)
        self.assertEqual(mse, 8.0)    

    def test_RMSE(self):
        """
        Test RMSE
        """
        # two inputs
        x1 = np.array([1, 4, 4, 4])
        x2 = np.array([4, 1, 1, 1])
        rmse = pa.misc.RMSE(x1, x2)
        self.assertEqual(rmse, 3.0)
        # one input
        e = x1 - x2
        rmse = pa.misc.RMSE(e)
        self.assertEqual(rmse, 3.0)    

    def test_MAE(self):
        """
        Test MAE
        """
        # two inputs
        x1 = np.array([2, 4, 4])
        x2 = np.array([4, 6, 2])
        mae = pa.misc.MAE(x1, x2)
        self.assertEqual(mae, 2.0)        
        # one input
        e = x1 - x2
        mae = pa.misc.MAE(e)
        self.assertEqual(mae, 2.0) 

    def test_logSE(self):
        """
        Test logSE
        """
        # two inputs
        x1 = np.array([2, 4, 4, 1, 2, 3, -1, 4])
        x2 = np.array([4, 6, 2, -1, 3, 4, 1, 3])
        logse = pa.misc.logSE(x1, x2)
        self.assertEqual(np.round(logse.sum(), 4), 30.1030)  
        # one input
        e = x1 - x2
        logse = pa.misc.logSE(e)
        self.assertEqual(np.round(logse.sum(), 4), 30.1030)  
 
        
        
        

