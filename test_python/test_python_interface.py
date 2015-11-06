import numpy as np
import gaussian_process_regression
import unittest

class BasicTest(unittest.TestCase):
    def setUp(self):
        self.gpr = gaussian_process_regression.GaussianProcessRegression(2, 1)

    def test_add_data_one_by_one(self):
        inp, outp = np.random.randn(2,1), np.random.randn(1)
        self.gpr.AddTrainingData()
        

        


gp = gpr.GaussianProcessRegression(2, 1)
inp, outp = np.random.randn(2,1), np.random.randn(1)
print inp, outp
gp.AddTrainingData(inp,outp)
gp.AddTrainingData(inp,outp)
gp.AddTrainingData(inp,outp)
