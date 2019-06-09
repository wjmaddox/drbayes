import unittest
import torch
import numpy as np

from swag.models.regression_net import SplitDim

class Test_SplitDim(unittest.TestCase):
    def test_splitdim(self, batch_size = 100, seed = 1, **kwargs):
        torch.random.manual_seed(seed)
        input = torch.randn(batch_size, 2)

        layer = SplitDim(1, nonlin_type=torch.nn.functional.softplus, correction=False)
        output = layer(input)

        self.assertLess(torch.norm(input[:,0] - output[:,0]), 1e-7)

        transformed_input = torch.nn.functional.softplus(input[:,1])

        relative_error = torch.norm(transformed_input - output[:,1])
        print(relative_error)
        self.assertLess(relative_error, 1e-5)

    def test_multiple_splitdim(self, batch_size = 100, seed = 1, dim = 4, full_dim = 7, **kwargs):
        torch.random.manual_seed(seed)
        input = torch.randn(batch_size, full_dim)

        layer = SplitDim(dim, nonlin_type=torch.nn.functional.softplus, correction=False)
        output = layer(input)

        for i in range(dim):
            err = torch.norm(input[:,i] - output[:,i])
            print(err)
            self.assertLess(err, 1e-7)

        for i in range(dim+1, full_dim):
            err = torch.norm(input[:,i] - output[:,i])
            print(err)
            self.assertLess(err, 1e-7)

        transformed_input = torch.nn.functional.softplus(input[:,dim])

        relative_error = torch.norm(transformed_input - output[:,dim])
        print(relative_error)
        self.assertLess(relative_error, 1e-5)        

if __name__ == "__main__":
    unittest.main()