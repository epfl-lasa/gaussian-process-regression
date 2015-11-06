import numpy as np
from gaussian_process_regression import GaussianProcessRegression
import unittest

class BasicTest(unittest.TestCase):
    def setUp(self):
        self.gpr = GaussianProcessRegression(2, 1)

    def test_add_data_one_by_one(self):
        inp, outp = np.random.randn(2,1), np.random.randn(1)
        self.gpr.AddTrainingData()
        

        



if __name__ == '__main__':
    unittest.main()
