import numpy as np
from gaussian_process_regression import GaussianProcessRegression
import unittest

class BasicTest(unittest.TestCase):
    def setUp(self):
        self.input_dim = 3
        self.output_dim = 2
        self.gpr = GaussianProcessRegression(self.input_dim,self.output_dim)

    def test_add_data_one_by_one_2darray(self):
        # as 2d-array
        inp = np.random.randn(self.input_dim, 1)
        outp = np.random.randn(self.output_dim, 1)
        self.gpr.AddTrainingData(inp, outp)
        self.assertTrue(np.allclose(inp, self.gpr.get_input_data()))
        self.assertTrue(np.allclose(outp, self.gpr.get_output_data()))

    def test_add_data_one_by_one_1darray(self):
        # as 1d-array
        inp = np.random.randn(self.input_dim)
        outp = np.random.randn(self.output_dim)
        self.gpr.AddTrainingData(inp, outp)
        self.assertTrue(np.allclose(inp,
                                    np.squeeze(self.gpr.get_input_data())))
        self.assertTrue(np.allclose(outp,
                                    np.squeeze(self.gpr.get_output_data())))

    def test_add_data_batch(self):
        n_data = 10
        inp = np.random.randn(self.input_dim, n_data)
        outp = np.random.randn(self.output_dim, n_data)
        self.gpr.AddTrainingData(inp, outp)
        print inp
        print self.gpr.get_input_data()

        self.assertTrue(np.allclose(inp, self.gpr.get_input_data()))
        self.assertTrue(np.allclose(outp, self.gpr.get_output_data()))


if __name__ == '__main__':
    unittest.main()
