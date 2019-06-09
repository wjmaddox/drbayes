import argparse
import os, sys
import math
import time
import tabulate

import torch
import tqdm
import torch.nn.functional as F
from torch import nn, distributions

import numpy as np

from swag import data, models, utils, losses
from swag.posteriors.vinf_model import VINFModel, ELBO_NF
from swag.posteriors.realnvp import RealNVP, construct_flow
from swag.posteriors import SWAG
from swag.posteriors.proj_model import Subspace
#from swag.posteriors.vi_model import VIModel, ELBO


def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.sum(np.log(ps))
    return nll


parser = argparse.ArgumentParser(description='Subspace VI')
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
parser.add_argument('--init_std', type=float, default=1.0, help='initial std of the variational distribution')

parser.add_argument('--checkpoint', type=str, default=None, required=True, help='path to SWAG checkpoint')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--eval_ensemble', action='store_true', help='store schedule')


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

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
    split_classes=args.split_classes
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
print("SWA:", utils.eval(loaders["train"], swag_model, criterion=losses.cross_entropy))

mean, var, cov_factor = swag_model.get_space()
subspace = Subspace(mean, cov_factor)

print(torch.norm(cov_factor, dim=1))

nvp_flow = construct_flow(cov_factor.shape[0], device=torch.cuda.current_device())

vi_model = VINFModel(
    base=model_cfg.base,
    subspace=subspace,
    flow=nvp_flow,
    prior_log_sigma=math.log(args.prior_std) + math.log(args.temperature) / 2,
    num_classes=num_classes,
    *model_cfg.args,
    **model_cfg.kwargs
)

vi_model = vi_model.cuda()
print(utils.eval(loaders["train"], vi_model, criterion=losses.cross_entropy))

elbo = ELBO_NF(losses.cross_entropy_output, len(loaders["train"].dataset), args.temperature)

optimizer = torch.optim.Adam([param for param in vi_model.parameters()], lr=0.01)

printf, logfile = utils.get_logging_print(os.path.join(args.dir, args.log_fname + '-%s.txt'))
print('Saving logs to: %s' % logfile)
columns = ['ep', 'acc', 'loss', 'kl', 'nll']

for epoch in range(args.epochs):
    train_res = utils.train_epoch(loaders['train'], vi_model, elbo, optimizer)
    values = ['%d/%d' % (epoch + 1, args.epochs), train_res['accuracy'], train_res['loss'],
              train_res['stats']['kl'], train_res['stats']['nll']]
    if epoch == 0:
        printf(tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f'))
    else:
        printf(tabulate.tabulate([values], columns, tablefmt='plain', floatfmt='8.4f').split('\n')[1])

utils.save_checkpoint(
    args.dir,
    epoch,
    name='vi_rnvp',
    state_dict=vi_model.state_dict()
)

if args.eval_ensemble:

    num_samples = 30

    predictions = np.zeros((len(loaders['test'].dataset), num_classes))
    targets = np.zeros(len(loaders['test'].dataset))
    
    printf, logfile = utils.get_logging_print(os.path.join(args.dir, args.log_fname + '-%s.txt'))
    print('Saving logs to: %s' % logfile)
    columns = ['iter ens', 'acc', 'nll']
    
    for i in range(num_samples):
        # utils.bn_update(loaders['train'], model, subset=args.bn_subset)
        vi_model.eval()
        k = 0
        for input, target in tqdm.tqdm(loaders['test']):
            input = input.cuda(non_blocking=True)
            torch.manual_seed(i)
    
            output = vi_model(input)
    
            with torch.no_grad():
                predictions[k:k+input.size()[0]] += F.softmax(output, dim=1).cpu().numpy()
            targets[k:(k+target.size(0))] = target.numpy()
            k += input.size()[0]
    
        values = ['%d/%d' % (i + 1, num_samples), np.mean(np.argmax(predictions, axis=1) == targets), nll(predictions / (i+1), targets)]
        if i == 0:
            printf(tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f'))
        else:
            printf(tabulate.tabulate([values], columns, tablefmt='plain', floatfmt='8.4f').split('\n')[1])

    predictions /= num_samples
