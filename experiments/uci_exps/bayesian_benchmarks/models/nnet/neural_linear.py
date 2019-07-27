import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from subspace_inference import utils
from bayesian_benchmarks.models.template import RegressionModel

def adjust_learning_rate(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor

class NLRegressionRunner(RegressionModel):
    def __init__(self, base, epochs, criterion, 
        batch_size = 50, lr_init=1e-2, momentum = 0.9, wd=1e-4,
        use_cuda = False, double_bias_lr=False, model_variance=True,
        const_lr=False, *args, **kwargs):
        
        self.model = base(*args, **kwargs)
        if use_cuda:
            self.model.cuda()

        self.use_cuda = use_cuda

        if not double_bias_lr:
            pars = self.model.parameters()
        else:
            pars = []
            for name, module in self.model.named_parameters():
                if 'bias' in str(name):
                    print('Doubling lr of ', name)
                    pars.append({'params':module, 'lr':2.0 * lr_init})
                else:
                    pars.append({'params':module, 'lr':lr_init})
       
        self.optimizer = torch.optim.SGD(pars, lr=lr_init, momentum=momentum, weight_decay=wd)

        self.const_lr = const_lr
        self.batch_size = batch_size

        # TODO: set up criterions better for classification
        if model_variance:
            self.criterion = criterion(noise_var = None)
        else:
            self.criterion = criterion(noise_var = 1.0)

        if self.criterion.noise_var is not None:
            self.var = self.criterion.noise_var

        self.epochs = epochs

        self.lr_init = lr_init

    def train(self, model, loader, optimizer, criterion, lr_init=1e-2, epochs=3000, 
        print_freq=100, use_cuda=False, const_lr=False):
        # copied from pavels regression notebook
        if const_lr:
            lr = lr_init

        train_res_list = []
        for epoch in range(epochs):
            if not const_lr:
                t = (epoch + 1) / epochs
                lr_ratio = 0.05
                
                if t <= 0.5:
                    factor = 1.0
                elif t <= 0.9:
                    factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
                else:
                    factor = lr_ratio

                lr = factor * lr_init
                adjust_learning_rate(optimizer, factor)
            
            train_res = utils.train_epoch(loader, model, criterion, optimizer, cuda=use_cuda, regression=True)
            train_res_list.append(train_res)
            
            if (epoch % print_freq == 0 or epoch == epochs - 1):
                print('Epoch %d. LR: %g. Loss: %.4f' % (epoch, lr, train_res['loss']))

        return train_res_list

   
    def fit(self, features, labels):
        self.features, self.labels = torch.FloatTensor(features), torch.FloatTensor(labels)

        # construct data loader 
        self.data_loader = DataLoader(TensorDataset(self.features, self.labels), batch_size = self.batch_size, shuffle = True)

        # now train with pre-specified options
        result = self.train(model=self.model, loader=self.data_loader, optimizer=self.optimizer, criterion=self.criterion, 
                lr_init=self.lr_init, use_cuda=self.use_cuda, epochs=self.epochs, const_lr=self.const_lr)

        if self.criterion.noise_var is not None:
            # another forwards pass through network to estimate noise variance
            preds, targets = utils.predictions(model=self.model, test_loader=self.data_loader, regression=True,cuda=self.use_cuda)
            self.var = np.power(np.linalg.norm(preds - targets), 2.0) / targets.shape[0]
            print(self.var)

        if self.use_cuda:
            self.features = self.features.cuda()
        self.net_features = self.model(self.features, output_features=True)
        return result

    def compute_posterior(self, x, y, eta = 6., l = 0.25):
        #print(x.size(), y.size())
        # see equations 1 and 2 in https://openreview.net/pdf?id=SyYe6k-CW
        # hypers from: https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits
        sigma_inv = x.t().matmul(x) + l * torch.eye(x.size(1), dtype = x.dtype, device = x.device)
        sigma = torch.inverse(sigma_inv)

        mu = sigma.matmul(x.t()).matmul(y)

        a = eta + x.size(0) / 2
        b = eta + 0.5 * (y.t().matmul(y) - mu.t().matmul(sigma_inv).matmul(mu))
        return mu, sigma, a, b.item()

    def predict(self, inputs):
        with torch.no_grad():
            # predictions from eq 20 of http://www.biostat.umn.edu/~ph7440/pubh7440/BayesianLinearModelGoryDetails.pdf
            inputs = torch.FloatTensor(inputs)
            if self.use_cuda:
                inputs = inputs.cuda()
            features = self.model(inputs, output_features = True).cpu().numpy()

            mu, sigma, a, b = self.compute_posterior(self.net_features.cpu(), self.labels.cpu())

            mean = np.dot(features, mu.numpy())

            # shape matrix of multivariate t
            shape = b / a * np.transpose(np.eye(features.shape[0]) + np.dot(features, sigma.numpy()) @ features.T)

            # return mean and variance
            return mean, (2*a / (2*a - 2)) * np.diag(shape)