import numpy as np

from numpy.random import multivariate_normal
from scipy.stats import norm

import torch
import math
from swag.losses import GaussianLikelihood

import unittest

class Test_GaussianLikelihood(unittest.TestCase):
    def test_scale_likelihood(self, n = 500, d = 1, seed = 1, **kwargs):
        np.random.seed(seed)

        # set mean, target, scale
        mean = np.random.randn(n, d)
        target = np.random.randn(n, d)
        scale = np.abs(np.random.randn(n,d))

        # join input
        input = np.hstack([mean, scale ** 2.0])

        # construct criterion
        criterion = GaussianLikelihood(noise_var=None)

        model = lambda x: x
        
        # calculate loss
        loss, output, mse_dict = criterion(model, torch.FloatTensor(input), torch.FloatTensor(target))

        # compute exact pdf
        numpy_pdf = norm.logpdf(target, loc=mean, scale=scale)
        print(loss.numpy(), numpy_pdf.mean() + 0.5 * math.log(2.0 * math.pi))
        self.assertLess(loss.numpy() + numpy_pdf.mean() - 0.5 * math.log(2.0 * math.pi), 1e-4)

if __name__ == "__main__":
    unittest.main()