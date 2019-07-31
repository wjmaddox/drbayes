import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from subspace_inference import utils
from subspace_inference.posteriors.proj_model import SubspaceModel
from subspace_inference.posteriors import SWAG, EllipticalSliceSampling, BenchmarkVIModel
from subspace_inference.posteriors import BenchmarkVINFModel
from subspace_inference.posteriors.realnvp import RealNVP, construct_flow

import copy

#from pyro.infer.mcmc import NUTS
#import pyro.distributions as dist

from bayesian_benchmarks.models.template import RegressionModel

def adjust_learning_rate(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor
    #return lr


class RegressionRunner(RegressionModel):
    def __init__(self, base, epochs, criterion, 
        batch_size = 50, lr_init=1e-2, momentum = 0.9, wd=1e-4,
        swag_lr = 1e-3, swag_freq = 1, swag_start = 50, subspace_type='pca', subspace_kwargs={'max_rank': 20},
        use_cuda = False, use_swag = False, double_bias_lr=False, model_variance=True,
        num_samples = 30, scale = 0.5, const_lr=False, *args, **kwargs):
        
        self.base = base
        self.model = base(*args, **kwargs)
        num_pars = 0
        for p in self.model.parameters():
            num_pars += p.numel()
        print('number of parameters: ', num_pars)
        
        if use_cuda:
            self.model.cuda()

        if use_swag:
            self.swag_model = SWAG(base, subspace_type=subspace_type, subspace_kwargs=subspace_kwargs, *args, **kwargs)
            if use_cuda:
                self.swag_model.cuda()
        else:
            self.swag_model = None

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

        self.use_swag = use_swag
        self.swag_start = swag_start
        self.swag_lr = swag_lr
        self.swag_freq = swag_freq
       
        self.num_samples = num_samples
        self.scale = scale

    def train(self, model, loader, optimizer, criterion, lr_init=1e-2, epochs=3000, 
        swag_model=None, swag=False, swag_start=2000, swag_freq=50, swag_lr=1e-3,
        print_freq=100, use_cuda=False, const_lr=False):
        # copied from pavels regression notebook
        if const_lr:
            lr = lr_init

        train_res_list = []
        for epoch in range(epochs):
            if not const_lr:
                t = (epoch + 1) / swag_start if swag else (epoch + 1) / epochs
                lr_ratio = swag_lr / lr_init if swag else 0.05
                
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
            if swag and epoch > swag_start:
                swag_model.collect_model(model)
            
            if (epoch % print_freq == 0 or epoch == epochs - 1):
                print('Epoch %d. LR: %g. Loss: %.4f' % (epoch, lr, train_res['loss']))

        return train_res_list

   
    def fit(self, features, labels):
        self.features, self.labels = torch.FloatTensor(features), torch.FloatTensor(labels)

        # construct data loader
	# may want to turn shuffle = False for the very smallest datasets (e.g. uci small) 
        self.data_loader = DataLoader(TensorDataset(self.features, self.labels), batch_size = self.batch_size, shuffle = True)

        # now train with pre-specified options
        result = self.train(model=self.model, loader=self.data_loader, optimizer=self.optimizer, criterion=self.criterion, 
                lr_init=self.lr_init, swag_model=self.swag_model, swag=self.use_swag, swag_start=self.swag_start,
                swag_freq=self.swag_freq, swag_lr=self.swag_lr, use_cuda=self.use_cuda, epochs=self.epochs, const_lr=self.const_lr)

        if self.criterion.noise_var is not None:
            # another forwards pass through network to estimate noise variance
            preds, targets = utils.predictions(model=self.model, test_loader=self.data_loader, regression=True,cuda=self.use_cuda)
            self.var = np.power(np.linalg.norm(preds - targets), 2.0) / targets.shape[0]
            print(self.var)

        return result

    def predict(self, features, swag_model=None):
        """
        default prediction method is to use built in Low rank Gaussian
        SWA: scale = 0.0, num_samples = 1
        """
        swag_model = swag_model if swag_model is not None else self.swag_model

        if self.use_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        with torch.no_grad():

            if swag_model is None:
                self.model.eval()
                preds = self.model(torch.FloatTensor(features).to(device)).data.cpu()

                if preds.size(1)==1:
                    var = torch.ones_like(preds[:,0]).unsqueeze(1) * self.var
                else:
                    var = preds[:,1].view(-1,1)
                    preds = preds[:,0].view(-1,1)

                print(var.mean())

            else:
                prediction = 0
                sq_prediction = 0
                for _ in range(self.num_samples):
                    swag_model.sample(scale=self.scale)
                    current_prediction = swag_model(torch.FloatTensor(features).to(device)).data.cpu()
                    prediction += current_prediction
                    if current_prediction.size(1) == 2:
                        #convert to standard deviation
                        current_prediction[:,1] = current_prediction[:,1] ** 0.5

                    sq_prediction += current_prediction ** 2.0
                # preds = bma/(self.num_samples)

                # compute mean of prediction
                # \mu^*
                preds = (prediction[:,0]/self.num_samples).view(-1,1)

                # 1/M \sum(\sigma^2(x) + \mu^2(x)) - \mu*^2
                var = torch.sum(sq_prediction, 1, keepdim=True)/self.num_samples - preds.pow(2.0)

                # add variance if not heteroscedastic
                if prediction.size(1)==1:
                    var = var + self.var
                    
            return preds.numpy(), var.numpy()


#class PyroRegRunner(RegressionRunner):
#    def __init__(self, base, epochs, criterion, 
#        batch_size = 50, lr_init=1e-2, momentum = 0.9, wd=1e-4,
#        swag_lr = 1e-3, swag_freq = 1, swag_start = 50, subspace_type='pca', subspace_kwargs={'max_rank': 20},
#        use_cuda = False, use_swag = True, model_variance=True,
#        num_samples = 30, scale = 0.5, double_bias_lr=False, const_lr=False,
#        prior_log_sigma = 1.0, kernel = NUTS, kernel_kwargs={'step_size':10},*args, **kwargs):
#
#        super(PyroRegRunner, self).__init__(base=base, epochs=epochs, criterion=criterion, model_variance=model_variance,
#            batch_size = batch_size, lr_init=lr_init, momentum=momentum, wd=wd, use_cuda = use_cuda, use_swag = use_swag,
#            swag_lr = swag_lr, swag_freq = swag_freq, swag_start = swag_start, subspace_type=subspace_type, subspace_kwargs=subspace_kwargs,
#            num_samples = num_samples, scale = scale, double_bias_lr=double_bias_lr, const_lr=const_lr, *args, **kwargs)
#
#        self.prior_log_sigma = prior_log_sigma
#        self.kernel = kernel
#        self.kernel_kwargs = kernel_kwargs
#
#        self.base = base
#        self.args = args
#        self.kwargs = kwargs
#
#        if self.criterion.noise_var is None:
#            self.likelihood = lambda x: dist.Normal(x[:,0], x[:,1])
#        else:
#            self.likelihood = lambda x: dist.Normal(x, self.criterion.noise_var)
#
#    def fit(self, features, labels):
#        # tran standard SWAG model
#        results = super().fit(features, labels)
#
#        mean, _, cov = self.swag_model.get_space()
#        if self.use_cuda:
#            mean, cov = mean.cuda(), cov.cuda()
#        subspace = SubspaceModel(mean, cov)
#
#        if self.use_cuda:
#            torch.set_default_tensor_type(torch.cuda.FloatTensor)
#
#        # now form BenchmarkPyroClass
#        self.bench_pyro = BenchmarkPyroModel(self.base, subspace, likelihood_given_outputs=self.likelihood, 
#            prior_log_sigma=self.prior_log_sigma, batch_size = self.batch_size, num_samples=self.num_samples, 
#            *self.args, **self.kwargs)
#        if self.use_cuda:
#            self.bench_pyro.cuda()
#            self.features, self.labels = self.features.cuda(), self.labels.cuda()
#        
#        self.bench_pyro.fit(self.features, self.labels)
#
#    def predict(self, features):
#        return super().predict(features, swag_model=self.bench_pyro)



class VIRegRunner(RegressionRunner):
    def __init__(self, base, epochs, criterion, 
        batch_size = 50, lr_init=1e-2, momentum = 0.9, wd=1e-4,
        swag_lr = 1e-3, swag_freq = 1, swag_start = 50, subspace_type='pca', subspace_kwargs={'max_rank': 20},
        use_cuda = False, use_swag = True,
        num_samples = 30, scale = 0.5, double_bias_lr=False, const_lr=False,
        prior_log_sigma = 1.0, temperature=1.0, model_variance=True, *args, **kwargs):

        super(VIRegRunner, self).__init__(base=base, epochs=epochs, criterion=criterion, 
            batch_size = batch_size, lr_init=lr_init, momentum=momentum, wd=wd, use_cuda = use_cuda, use_swag = use_swag,
            swag_lr = swag_lr, swag_freq = swag_freq, swag_start = swag_start, subspace_type=subspace_type, subspace_kwargs=subspace_kwargs,
            num_samples = num_samples, scale = scale, double_bias_lr=double_bias_lr, const_lr=const_lr, model_variance=model_variance,
            *args, **kwargs)

        self.base = base
        self.args = args
        self.kwargs = kwargs
        self.prior_log_sigma = prior_log_sigma
        self.temperature = temperature
        self.use_cuda = use_cuda

    def fit(self, features, labels):
            # tran standard SWAG model
            results = super().fit(features, labels)

            mean, _, cov = self.swag_model.get_space()
            if self.use_cuda:
                mean, cov = mean.cuda(), cov.cuda()
            subspace = SubspaceModel(mean, cov)

            if self.use_cuda:
                torch.set_default_tensor_type(torch.cuda.FloatTensor)
            
            if self.temperature is None:
                self.temperature = features.shape[0] / cov.size(0)
                print('Temperature:', self.temperature)

            self.bench_vi = BenchmarkVIModel(loader=self.data_loader, criterion=self.criterion, epochs=self.num_samples, 
                        base = self.base, subspace = subspace, prior_log_sigma=self.prior_log_sigma, temperature=self.temperature,
                        use_cuda=self.use_cuda, num_samples=features.shape[0], *self.args, **self.kwargs)
            
            self.bench_vi.fit()

    def predict(self, features):
        return super().predict(features, swag_model=self.bench_vi)



class ESSRegRunner(RegressionRunner):
    def __init__(self, base, epochs, criterion, 
        batch_size = 50, lr_init=1e-2, momentum = 0.9, wd=1e-4,
        swag_lr = 1e-3, swag_freq = 1, swag_start = 50, subspace_type='pca', subspace_kwargs={'max_rank': 20},
        use_cuda = False, use_swag = False, const_lr = False, double_bias_lr = False,model_variance=True,
        num_samples = 30, scale = 0.5, temperature = 1., mb = False, prior_log_sigma = 1.0, *args, **kwargs):
    
        super(ESSRegRunner, self).__init__(base=base, epochs=epochs, criterion=criterion, model_variance=model_variance,
            batch_size = batch_size, lr_init=lr_init, momentum=momentum, wd=wd, use_cuda = use_cuda, use_swag = use_swag,
            swag_lr = swag_lr, swag_freq = swag_freq, swag_start = swag_start, subspace_type=subspace_type, subspace_kwargs=subspace_kwargs,
            num_samples = num_samples, scale = scale, const_lr=const_lr, double_bias_lr=double_bias_lr, *args, **kwargs)

        self.prior_std = np.exp(prior_log_sigma)
        self.temperature = temperature
        self.mb = mb
        self.args = args
        self.kwargs = kwargs
        self.use_cuda = use_cuda

    def fit(self, features, labels):
        # tran standard SWAG model
        results = super().fit(features, labels)

        mean, var, cov = self.swag_model.get_space()
        if self.use_cuda:
            mean, cov = mean.cuda(), cov.cuda()

        if self.temperature is None:
            self.temperature = features.shape[0] / cov.size(0)
            print('Temperature:', self.temperature)

        # print(cov.size())
        subspace = SubspaceModel(mean, cov)
        self.ess_model = EllipticalSliceSampling(self.base, subspace=subspace, var=var, loader=self.data_loader, 
                                    criterion=self.criterion, num_samples=self.num_samples, use_cuda = self.use_cuda,
                                    *self.args, **self.kwargs)
        train_loss = self.ess_model.fit(scale=self.prior_std, use_cuda=self.use_cuda, temperature = self.temperature, minibatch=self.mb)
        print(train_loss)

    def predict(self, features):
        return super().predict(features, swag_model=self.ess_model)


#class RealNVPRegRunner(RegressionRunner):
#    def __init__(self, base, epochs, criterion, 
#        batch_size=50, lr_init=1e-2, momentum=0.9, wd=1e-4, swag_lr=1e-3, swag_freq=1,
#        swag_start=50, subspace_type='pca', subspace_kwargs={'max_rank': 20}, use_cuda=False,
#        use_swag=True, num_samples=30, scale=0.5, prior_log_sigma=1.0,
#        kwargs_flow={}, *args, **kwargs):
#
#        super(RealNVPRegRunner, self).__init__(base=base, epochs=epochs, criterion=criterion, 
#            batch_size=batch_size, lr_init=lr_init, momentum=momentum, wd=wd, use_cuda=use_cuda, use_swag=use_swag,
#            swag_lr=swag_lr, swag_freq=swag_freq, swag_start=swag_start, subspace_type=subspace_type, subspace_kwargs=subspace_kwargs,
#            num_samples=num_samples, scale=scale, *args, **kwargs)
#
#        self.prior_log_sigma = prior_log_sigma
#
#        self.base = base
#        self.args = args
#        self.kwargs = kwargs
#        self.kwards_flow = kwargs_flow
#
#    def fit(self, features, labels):
#        # train standard SWAG model
#        results = super().fit(features, labels)
#
#        mean, _, cov = self.swag_model.get_space()
#        if self.cuda:
#            mean, cov = mean.cuda(), cov.cuda()
#        subspace = Subspace(mean, cov)
#
#        torch.set_default_tensor_type(torch.cuda.FloatTensor)
#
#        # form BenchmarkVINFModel
#        nvp_flow = construct_flow(self.features.size(1), **self.kwargs_flow)
#        self.bench_model = BenchmarkVINFModel(self.data_loader, self.criterion, self.optimizer, self.epochs,
#            self.base, subspace, nvp_flow, self.prior_log_sigma, lr, len(self.data_loader))
#        if self.cuda:
#            self.bench_pyro.cuda()
#            self.features, self.labels = self.features.cuda(), self.labels.cuda()
#        
#        self.bench_pyro.fit(self.features, self.labels)
#
#    def predict(self, features):
#        return super().predict(features, swag_model=self.bench_model)

