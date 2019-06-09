import argparse
import os, sys
import math
import time
import tabulate
import copy

import torch

import numpy as np

from swag import data, models, utils, losses
from swag.posteriors import SWAG
from swag.posteriors.proj_model import ProjectedModel


parser = argparse.ArgumentParser(description='Subspace MAP')
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

parser.add_argument('--rank', type=int, default=2, metavar='N', help='approximation rank (default: 2')
parser.add_argument('--subspace', choices=['covariance', 'pca', 'freq_dir'], default='covariance')

parser.add_argument('--prior_std', type=float, default=20.0, help='std of the prior distribution')
parser.add_argument('--lr', type=float, default=1e-4, help='initial std of the variational distribution')

#parser.add_argument('--checkpoint', type=str, default=None, required=True, help='path to SWAG checkpoint')

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

model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()

swag_model = SWAG(
    model_cfg.base,
    num_classes=num_classes,
    subspace_type='pca',
    subspace_kwargs={
        'max_rank': 140,
        'pca_rank': args.rank,
    },
    *model_cfg.args,
    **model_cfg.kwargs
)
swag_model.to(args.device)

def checkpoint_num(filename):
    num = filename.split("-")[1]
    num = num.split(".")[0]
    num = int(num)
    return num

for file in os.listdir(args.dir):
    if "checkpoint" in file and checkpoint_num(file) > 160:
        path = os.path.join(args.dir, file)
        print('Loading %s' % path)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        #W.append(np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()]))
        swag_model.collect_model(model)

#print('Loading: %s' % args.checkpoint)
#ckpt = torch.load(args.checkpoint)
#swag_model.load_state_dict(ckpt['state_dict'], strict=False)

swag_model.set_swa()


mean, var, subspace = swag_model.get_space()
mean = mean.cuda()
subspace = subspace.cuda()

proj_params = torch.zeros(subspace.size(0), 1, dtype = subspace.dtype, device = subspace.device, requires_grad = True)
print(proj_params.device, subspace.device)
proj_model = ProjectedModel(model=copy.deepcopy(model).cuda(), mean=mean.unsqueeze(1),  projection=subspace, proj_params=proj_params)

def criterion(model, input, target, scale=args.prior_std):
    likelihood, output, _ = losses.cross_entropy(model, input, target)
    prior = 1/(scale ** 2.0 * input.size(0)) * proj_params.norm()
    return likelihood + prior, output, {'nll':likelihood * input.size(0), 'prior':proj_params.norm()}

optimizer = torch.optim.SGD([proj_params], lr = 5e-4, momentum = 0.9, weight_decay=0)

swag_model.sample(0)
utils.bn_update(loaders['train'], swag_model)
print( utils.eval(loaders['test'], swag_model, criterion) )

printf, logfile = utils.get_logging_print(os.path.join(args.dir, args.log_fname + '-%s.txt'))
print('Saving logs to: %s' % logfile)
#printf=print
columns = ['ep', 'acc', 'loss', 'prior']

for epoch in range(args.epochs):
    train_res = utils.train_epoch(loaders['train'], proj_model, criterion, optimizer)
    values = ['%d/%d' % (epoch + 1, args.epochs), train_res['accuracy'], train_res['loss'],
              train_res['stats']['prior'], train_res['stats']['nll']]
    if epoch == 0:
        printf(tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f'))
    else:
        printf(tabulate.tabulate([values], columns, tablefmt='plain', floatfmt='8.4f').split('\n')[1])

print( utils.eval(loaders['test'], proj_model, criterion) )

utils.save_checkpoint(
    args.dir,
    epoch,
    name='projected',
    state_dict=proj_model.state_dict()
)
