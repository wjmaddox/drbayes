import torch
from torch.distributions.lowrank_multivariate_normal import _batch_lowrank_logdet

from swag.utils import flatten
from swag import utils

def compute_swag_param_norm(model):
    l2norm = 0
    
    #for module, name in model.base_model.parameters():
    #    w = getattr(module, name)
    #    l2norm += flatten(w).norm()
    for w in model.base_model.parameters():
        l2norm += flatten(w).norm()
    return l2norm

def compute_logdet(var, cov_mat):
    dist = torch.distributions.LowRankMultivariateNormal(torch.zeros_like(var), cov_mat, var)
    log_det = _batch_lowrank_logdet(dist._unbroadcasted_cov_factor,
                                dist._unbroadcasted_cov_diag,
                                dist._capacitance_tril)
    return log_det

def log_marginal_laplace(log_joint_swa, logdet, model_numparams, num_datapoints=50000):
    #compute number of parameters of model
    #model_numparams = compute_num_params(swa_model)

    #log(2pi) = 1.83
    # p/2 * log(2pi)
    normalizing_constant = model_numparams/2.0 * (1.8378770664093453 - num_datapoints)

    # 1/2 * log|\Sigma|
    logdet_term = 0.5 * logdet

    # p/2 * log(2pi) - 1/2 * log|\Sigma| + log(p(Y|X,\bar{\theta})p(\bar{\theta}))
    laplace_estimate = normalizing_constant + logdet_term + log_joint_swa

    return laplace_estimate

def compute_joint(loader, model, criterion, wd_scale=3e-4):
    nll_dict = utils.eval(loader=loader, model=model, criterion=criterion)
    nll = nll_dict['loss'] * len(loader.dataset)

    prior = compute_swag_param_norm(model) * wd_scale

    return nll + prior

def marginal_loglikelihood(model, loader, criterion, wd=3e-4, N=45000):
    
    log_joint = compute_joint(loader=loader, criterion=criterion, model=model, wd_scale=wd)

    _, var, subspace = model.get_space()
    num_parameters = len(var)

    
    subspace_evals, _ = torch.eig(subspace.matmul(subspace.t()))
    print(var.mean(), subspace_evals.sum())
    ll = torch.zeros(subspace.size(0) - 1)
    for rank in range(1,subspace.size(0)):
        updated_var = var + subspace_evals[rank:,0].sum() / (N - 1)
        logdet = compute_logdet(updated_var, subspace[:rank,:].t())
        ll[rank-1] = log_marginal_laplace(log_joint, logdet, (rank+1) * num_parameters, num_datapoints=N)
    print(ll)
    return torch.argmax(ll) + 1

