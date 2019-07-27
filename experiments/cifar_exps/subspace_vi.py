import argparse
import os, sys
import math
import time
import tabulate

import torch
import tqdm
import torch.nn.functional as F

import numpy as np

from subspace_inference import data, models, utils, losses
from subspace_inference.posteriors import SWAG
from subspace_inference.posteriors.vi_model import VIModel, ELBO
from subspace_inference.posteriors.proj_model import SubspaceModel

import sklearn.decomposition

def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.mean(np.log(ps + 1e-12))
    return nll


parser = argparse.ArgumentParser(description='Subspace VI')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
#parser.add_argument('--log_fname', type=str, default=None, required=True, help='file name for logging')

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

parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs (default: 50')
parser.add_argument('--num_samples', type=int, default=30, metavar='N', help='number of epochs (default: 30')

parser.add_argument('--temperature', type=float, default=1., required=True, 
                    metavar='N', help='Temperature (default: 1.')
parser.add_argument('--no_mu', action='store_true', help='Do not learn the mean of posterior')

parser.add_argument('--rank', type=int, default=2, metavar='N', help='approximation rank (default: 2')
parser.add_argument('--random', action='store_true')

parser.add_argument('--prior_std', type=float, default=1.0, help='std of the prior distribution')
parser.add_argument('--init_std', type=float, default=1.0, help='initial std of the variational distribution')

parser.add_argument('--checkpoint', type=str, default=None, required=True, help='path to SWAG checkpoint')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--bn_subset', type=float, default=1.0, help='BN subset for evaluation (default 1.0)')
parser.add_argument('--max_rank', type=int, default=20, help='maximum rank')

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
        'max_rank': args.max_rank,
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
print("SWA:", utils.eval(loaders["test"], swag_model, criterion=losses.cross_entropy))

mean, var, cov_factor = swag_model.get_space()

print(torch.norm(cov_factor, dim=1))
#print(var)

if args.random:
    scale = 0.5 * (np.linalg.norm(cov_factor[1, :]) + np.linalg.norm(cov_factor[0, :]))
    print(scale)
    np.random.seed(args.seed)
    cov_factor = np.random.randn(*cov_factor.shape)


    tsvd = sklearn.decomposition.TruncatedSVD(n_components=args.rank, n_iter=7, random_state=args.seed)
    tsvd.fit(cov_factor)

    cov_factor = tsvd.components_
    cov_factor /= np.linalg.norm(cov_factor, axis=1, keepdims=True)
    cov_factor *= scale

    print(cov_factor[:, 0])

    cov_factor = torch.FloatTensor(cov_factor, device=mean.device)

vi_model = VIModel(
    subspace=SubspaceModel(mean.cuda(), cov_factor.cuda()),
    init_inv_softplus_sigma=math.log(math.exp(args.init_std) - 1.0),
    prior_log_sigma=math.log(args.prior_std),
    num_classes=num_classes,
    base=model_cfg.base,
    with_mu=not args.no_mu,
    *model_cfg.args,
    **model_cfg.kwargs
)

vi_model = vi_model.cuda()
print(utils.eval(loaders["train"], vi_model, criterion=losses.cross_entropy))

elbo = ELBO(losses.cross_entropy, len(loaders['train'].dataset), args.temperature)

#optimizer = torch.optim.Adam([param for param in vi_model.parameters()], lr=0.01)
optimizer = torch.optim.SGD([param for param in vi_model.parameters()], lr=args.lr, momentum=0.9)

#printf, logfile = utils.get_logging_print(os.path.join(args.dir, args.log_fname + '-%s.txt'))
#print('Saving logs to: %s' % logfile)
columns = ['ep', 'acc', 'loss', 'kl', 'nll', 'sigma_1', 'time']

epoch = 0
for epoch in range(args.epochs):
    time_ep = time.time()
    train_res = utils.train_epoch(loaders['train'], vi_model, elbo, optimizer)
    time_ep = time.time() - time_ep
    sigma_1 = torch.nn.functional.softplus(vi_model.inv_softplus_sigma.detach().cpu())[0].item()
    values = ['%d/%d' % (epoch + 1, args.epochs), train_res['accuracy'], train_res['loss'],
              train_res['stats']['kl'], train_res['stats']['nll'], sigma_1, time_ep]
    if epoch == 0:
        print(tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f'))
    else:
        print(tabulate.tabulate([values], columns, tablefmt='plain', floatfmt='8.4f').split('\n')[1])



print("sigma:", torch.nn.functional.softplus(vi_model.inv_softplus_sigma.detach().cpu()))
if not args.no_mu:
    print("mu:", vi_model.mu.detach().cpu().data)

utils.save_checkpoint(
    args.dir,
    epoch,
    name='vi',
    state_dict=vi_model.state_dict()
)


eval_model = model_cfg.base(num_classes=num_classes, *model_cfg.args, **model_cfg.kwargs)
eval_model.to(args.device)

num_samples = args.num_samples

ens_predictions = np.zeros((len(loaders['test'].dataset), num_classes))
targets = np.zeros(len(loaders['test'].dataset))

#printf, logfile = utils.get_logging_print(os.path.join(args.dir, args.log_fname + '-%s.txt'))
#print('Saving logs to: %s' % logfile)
columns = ['iter ens', 'acc', 'nll']

for i in range(num_samples):
    with torch.no_grad():
        w = vi_model.sample()
        offset = 0
        for param in eval_model.parameters():
            param.data.copy_(w[offset:offset+param.numel()].view(param.size()).to(args.device))
            offset += param.numel()

    utils.bn_update(loaders['train'], eval_model, subset=args.bn_subset)

    pred_res = utils.predict(loaders['test'], eval_model)
    ens_predictions += pred_res['predictions']
    targets = pred_res['targets']

    values = ['%3d/%3d' % (i + 1, num_samples),
              np.mean(np.argmax(ens_predictions, axis=1) == targets),
              nll(ens_predictions / (i + 1), targets)]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if i == 0:
        print(table)
    else:
        print(table.split('\n')[2])

ens_predictions /= num_samples
ens_acc = np.mean(np.argmax(ens_predictions, axis=1) == targets)
ens_nll = nll(ens_predictions, targets)

np.savez(
    os.path.join(args.dir, 'ens.npz'),
    seed=args.seed,
    ens_predictions=ens_predictions,
    targets=targets,
    ens_acc=ens_acc,
    ens_nll=ens_nll
)
