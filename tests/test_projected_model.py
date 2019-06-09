import torch

import unittest
from swag.posteriors.proj_model import ProjectedModel

class TestProjModel(unittest.TestCase):
    def test_proj_model(self):
        
        model = torch.nn.Sequential(torch.nn.Linear(30,1, bias = False))
        z = torch.randn(10, 1, requires_grad = True)
        proj = torch.randn(30, 10)
        mean = torch.randn(30, 1)

        znew = z.data.clone()
        znew.requires_grad = True

        pm = ProjectedModel(proj_params = z, mean=mean, projection=proj, model=model)


        #to ensure no gradients accumulate over optimization

        for _ in range(10):
            unproj_model = mean + proj.matmul(znew)

            if z.grad is not None:
                z.grad.data.zero_()
            if znew.grad is not None:
                znew.grad.data.zero_()

            input = torch.randn(150, 30)
            label = torch.randn(150, 1)
            
            output = pm(input)
            loss = ( (output - label) ** 2.0 ).sum()
            loss.backward()

            projected_gradient = z.grad


            lossnew = ( (input.matmul(unproj_model) - label) ** 2.0 ).sum()
            lossnew.backward()
            nm_projected_gradient = znew.grad

            self.assertEqual((projected_gradient - nm_projected_gradient).norm().item(), 0.0)

if __name__ == "__main__":
    unittest.main()