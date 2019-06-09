import argparse
import os, sys
import math
import time
import tabulate
import torch
from pyro.infer.mcmc import HMC, MCMC, NUTS
import pyro.distributions as dist
import numpy as np
from itertools import islice
import tqdm
import torch.nn.functional as F

from swag import data, models, utils, losses
from swag.posteriors import SWAG
from swag.posteriors.pyro import PyroModel, TemperedCategorical
from swag.posteriors.proj_model import Subspace


def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.sum(np.log(ps))
    return nll


parser = argparse.ArgumentParser(description='Subspace MCMC Sampling')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--log_fname', type=str, default=None, required=True, help='file name for logging')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')

parser.add_argument('--use_test', dest='use_test', action='store_true',
                    help='use test dataset instead of validation (default: False)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs (default: 100')

parser.add_argument('--temperature', type=float, default=1., required=True, 
                    metavar='N', help='Temperature (default: 1.')
parser.add_argument('--no_mu', action='store_true', help='Do not learn the mean of posterior')

parser.add_argument('--rank', type=int, default=2, metavar='N', help='approximation rank (default: 2')
parser.add_argument('--subspace', choices=['covariance', 'pca', 'freq_dir'], default='covariance')

parser.add_argument('--prior_std', type=float, default=20.0, help='std of the prior distribution')

parser.add_argument('--checkpoint', type=str, default=None, required=True, help='path to SWAG checkpoint')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--no_schedule', action='store_true', help='store schedule')


args = parser.parse_args()

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

#torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    #model_cfg.transform_train,
    model_cfg.transform_test,
    model_cfg.transform_test,
    use_validation=not args.use_test,
    split_classes=args.split_classes,
    shuffle_train=False
)

print('Preparing model')
print(*model_cfg.args)

swag_model = SWAG(
    model_cfg.base,
    num_classes=num_classes,
    subspace_type='pca',
    subspace_kwargs={
        'max_rank': 20,
        'pca_rank': args.rank,
    },
    *model_cfg.args,
    **model_cfg.kwargs
)
swag_model.to(args.device)

print('Loading: %s' % args.checkpoint)
ckpt = torch.load(args.checkpoint)
swag_model.load_state_dict(ckpt['state_dict'], strict=False)

swag_model.set_swa()

mean, var, cov_factor = swag_model.get_space()

print(torch.norm(cov_factor, dim=1))

likelihood = lambda logits: TemperedCategorical(logits=logits, temperature=args.temperature)
#likelihood = lambda logits: dist.Categorical(logits=logits)

pyro_model = PyroModel(
    base=model_cfg.base,
    subspace = Subspace(mean.cuda(), cov_factor.cuda()),
    prior_log_sigma=math.log(args.prior_std), 
    likelihood_given_outputs=likelihood,
    num_classes=num_classes,
    *model_cfg.args,
    **model_cfg.kwargs
)
pyro_model.cuda()


inpts, trgts = [], []
for i, (inpt, trgt) in enumerate(loaders["train"]):
    inpts.append(inpt)
    trgts.append(trgt)

inpts = torch.cat(inpts).cuda()#[:25000]
trgts = torch.cat(trgts).cuda()#[:25000]
batch_size = 100#args.batch_size
num_batches = inpts.shape[0] // batch_size
print([num_batches, batch_size,*inpts.shape[1:]])
inpts = inpts.reshape([num_batches, batch_size, *inpts.shape[1:]])
trgts = trgts.reshape([num_batches, batch_size])
print("Inputs:", inpts.shape)
print("Targets:", trgts.shape)

printf, logfile = utils.get_logging_print(os.path.join(args.dir, args.log_fname + '-%s.txt'))
print('Saving logs to: %s' % logfile)

nuts_kernel = NUTS(pyro_model.model, step_size=10.)

num_samples = 30

# x_, y_ = loaders["train"].dataset.tensors
mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=10).run(inpts, trgts)
#mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=100).run(islice(loaders["train"], 1000))
samples = torch.cat(list(mcmc_run.marginal(sites="t").support(flatten=True).values()), dim=-1)
print(samples)

utils.save_checkpoint(
    args.dir,
    0,
    name='nuts',
    state_dict=pyro_model.state_dict()
)


predictions = np.zeros((len(loaders['test'].dataset), num_classes))
targets = np.zeros(len(loaders['test'].dataset))

printf, logfile = utils.get_logging_print(os.path.join(args.dir, args.log_fname + '-%s.txt'))
print('Saving logs to: %s' % logfile)
columns = ['iter ens', 'acc', 'nll']

for i in range(num_samples):
    # utils.bn_update(loaders['train'], model, subset=args.bn_subset)
    pyro_model.eval()
    k = 0
    pyro_model.t.set_(samples[i, :])
    for input, target in tqdm.tqdm(loaders['test']):
        input = input.cuda(non_blocking=True)
        torch.manual_seed(i)

        output = pyro_model(input)

        with torch.no_grad():
            predictions[k:k+input.size()[0]] += F.softmax(output, dim=1).cpu().numpy()
        targets[k:(k+target.size(0))] = target.numpy()
        k += input.size()[0]

    values = ['%d/%d' % (i + 1, num_samples), np.mean(np.argmax(predictions, axis=1) == targets), nll(predictions / (i+1), targets)]
    if i == 0:
        printf(tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f'))
    else:
        printf(tabulate.tabulate([values], columns, tablefmt='plain', floatfmt='8.4f').split('\n')[1])

pyro_model.t.set_(torch.zeros_like(pyro.model.t))
print(utils.eval(loaders["train"], pyro_model, criterion=losses.cross_entropy))

predictions /= num_samples
