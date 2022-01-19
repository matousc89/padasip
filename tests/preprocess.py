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
        x = pa.preprocess.standardize(u)
        self.assertAlmostEqual(x.std(), 1.0, 10)
        self.assertEqual(np.round(x.mean(), 3), -0.0)
        
    def test_standardize_back(self):
        """
        Test de-standardization.
        """
        x = range(1000)
        u = pa.preprocess.standardize_back(x, 2, 10)
        self.assertEqual(np.round(u.std(), 3), 2886.75)
        self.assertEqual(np.round(u.mean(), 3), 4997.0)
        
    def test_input_from_history(self):
        """
        Test input from history function.
        """
        u = range(1000)
        x = pa.preprocess.input_from_history(u, 4)       
        self.assertEqual(x.shape, (997, 4))
        self.assertEqual(np.round(np.array(u).mean(), 5), np.round(x.mean(), 5))
        
    def test_pca(self):
        """
        Principal Component Analysis
        """
        np.random.seed(100) 
        x = np.random.uniform(1, 10, (100, 3))
        # PCA components
        out = pa.preprocess.PCA_components(x) 
        self.assertEqual(np.round(np.array(out).mean(), 5), 6.82133)
        # PCA analysis
        out = pa.preprocess.PCA(x, 2) 
        self.assertEqual(out.shape, (100, 2))
        self.assertEqual(np.round(np.array(out).mean(), 5), 3.98888)

    def test_lda(self):
        """
        Linear Disciminant Analysis
        """
        np.random.seed(100) 
        N = 150 
        classes = np.array(["1", "a", 3]) 
        cols = 4
        x = np.random.random((N, cols)) # random data
        labels = np.random.choice(classes, size=N) # random labels
        # LDA components
        out = pa.preprocess.LDA_discriminants(x, labels)
        self.assertEqual(np.round(np.array(out).mean(), 5), 0.01298)
        # LDA analysis
        new_x = pa.preprocess.LDA(x, labels, n=2)  
        self.assertEqual(np.round(np.array(new_x).mean(), 5), -0.50907)
        self.assertEqual(new_x.shape, (150, 2))
        
        
        
        
        
        

