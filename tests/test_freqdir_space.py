import unittest
import numpy as np
from numpy.random import multivariate_normal
import scipy
from scipy.linalg import toeplitz
import torch

from swag.posteriors.subspaces import FreqDirSpace


def reconstruction_error(A, V):
    A_rec = np.dot(np.dot(A, V.T), V / (np.sum(np.square(V), axis=1, keepdims=True)))
    return np.mean(np.sum(np.square(A_rec - A), axis=1))


class TestFreqDirSpace(unittest.TestCase):
    def test_freqdir(self, n=200, d=10, rank=8, seed=1):
        np.random.seed(seed)
        
        # construct n random samples from N(0, d)
        cov_mat = toeplitz(np.abs(np.random.randn(d)))
        cov_mat = cov_mat + np.sqrt(20*d) * np.eye(d)
        A = multivariate_normal(np.zeros(d), cov_mat, n)

        fd = FreqDirSpace(num_parameters=d, max_rank=rank)

        # collect model
        for a in A:
            a_tensor = torch.FloatTensor(a)
            fd.collect_vector(a_tensor)
        
        subspace = fd.get_space().numpy()

        # ensure that the size of the matrix is correct
        self.assertEqual(subspace.shape[0], rank)
        self.assertEqual(subspace.shape[1], d)
        # ensure orthogonal matrix
        self.assertLess(np.max(np.abs(np.triu(subspace @ subspace.T, k=1))), 3e-5)

        [_, s, V] = scipy.linalg.svd(A / np.sqrt(n - 1), full_matrices=False)
        V = s[:rank // 2, None] * V[:rank // 2, :]

        fd_rec_err = reconstruction_error(A, subspace)
        V_rec_err = reconstruction_error(A, V)
        self.assertLessEqual(fd_rec_err, 2 * V_rec_err)

        A_cov = np.dot(A.T, A) / (n - 1)
        fd_cov = np.dot(subspace.T, subspace)
        V_cov = np.dot(V.T, V)
        self.assertLess(np.linalg.norm(A_cov - fd_cov),
                        2 * np.linalg.norm(A_cov - V_cov) ** 2 / rank)
        
    def test_mult_freqdir(self):
        for n in [300, 500, 1000]:
            for d in [10, 100]:
                for rank in [0.3, 0.5]:
                    self.test_freqdir(n, d, int(rank*d))


if __name__ == "__main__":
    unittest.main()
