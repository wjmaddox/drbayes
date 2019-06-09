import argparse
import tabulate
import os
import sys

import torch
import numpy as np

from swag import data, models, utils
from swag.posteriors import SWAG

import sklearn.decomposition

def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.mean(np.log(ps + 1e-12))
    return nll


parser = argparse.ArgumentParser(description='Subspace VI')

parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
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

parser.add_argument('--num_samples', type=int, default=30,
                    metavar='N', help='Number of samples in ensemble (default: 30)')

parser.add_argument('--scale', type=float, default=1., required=True,
                    metavar='S', help='Scale (default: 1.)')
parser.add_argument('--rank', type=int, default=2, metavar='N', help='approximation rank (default: 2)')
parser.add_argument('--random', action='store_true')

parser.add_argument('--checkpoint', type=str, default=None, required=True, help='path to SWAG checkpoint')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--bn_subset', type=float, default=1.0, help='BN subset for evaluation (default 1.0)')


args = parser.parse_args()

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

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

if args.random:
    mean, _, cov_factor = swag_model.get_space()
    cov_factor = cov_factor.cpu().numpy()
    print(np.linalg.norm(cov_factor, axis=1))
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

    swag_model.cov_factor.copy_(torch.FloatTensor(cov_factor, device=mean.device))

ens_predictions = np.zeros((len(loaders['test'].dataset), num_classes))
targets = np.zeros(len(loaders['test'].dataset))

columns = ['iter ens', 'acc', 'nll']

with torch.no_grad():
    for i in range(args.num_samples):
        swag_model.sample(scale=args.scale)

        utils.bn_update(loaders['train'], swag_model, subset=args.bn_subset)

        pred_res = utils.predict(loaders['test'], swag_model)
        ens_predictions += pred_res['predictions']
        targets = pred_res['targets']

        values = ['%3d/%3d' % (i + 1, args.num_samples),
                  np.mean(np.argmax(ens_predictions, axis=1) == targets),
                  nll(ens_predictions / (i + 1), targets)]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        if i == 0:
            print(table)
        else:
            print(table.split('\n')[2])

ens_predictions /= args.num_samples
np.savez(
    os.path.join(args.dir, 'ens.npz'),
    seed=args.seed,
    ens_predictions=ens_predictions,
    targets=targets,
)
