import argparse
import os, sys
import math
import time
import tabulate
import copy

import torch

import numpy as np

from subspace_inference import data, models, utils, losses
from subspace_inference.posteriors import SWAG
from subspace_inference.posteriors.proj_model import SubspaceModel
from subspace_inference.posteriors.elliptical_slice import elliptical_slice

import sklearn.decomposition

parser = argparse.ArgumentParser(description='Subspace ESS')
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

parser.add_argument('--num_samples', type=int, default=30, metavar='N', help='number of epochs (default: 30')

parser.add_argument('--curve', action='store_true')
parser.add_argument('--random', action='store_true')
parser.add_argument('--rank', type=int, default=2, metavar='N', help='approximation rank (default: 2')
parser.add_argument('--checkpoint', type=str, default=None, required=True, help='path to SWAG checkpoint')


parser.add_argument('--prior_std', type=float, default=1.0, help='std of the prior distribution')
parser.add_argument('--temperature', type=float, default=1., help='temperature')

parser.add_argument('--bn_subset', type=float, default=1.0, help='BN subset for evaluation (default 1.0)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--no_schedule', action='store_true', help='store schedule')


args = parser.parse_args()


def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.mean(np.log(ps + 1e-12))
    return nll

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'ess_command.sh'), 'w') as f:
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
    transform_train=model_cfg.transform_test,
    transform_test=model_cfg.transform_test,
    shuffle_train=False,
    use_validation=not args.use_test,
    split_classes=args.split_classes
)

loaders_bn, _ = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    transform_train=model_cfg.transform_train,
    transform_test=model_cfg.transform_test,
    shuffle_train=True,
    use_validation=not args.use_test,
    split_classes=args.split_classes
)

print('Preparing model')
print(*model_cfg.args)

model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)


if args.curve:

    assert args.rank == 2

    checkpoint = torch.load(args.checkpoint)
    num_parameters = sum([p.numel() for p in model.parameters()])
    w = np.zeros((3, num_parameters))

    for i in range(3):
        offset = 0
        for name, param in model.named_parameters():

            size = param.numel()

            if 'net.%s_1' % name in checkpoint['model_state']:
                w[i, offset:offset+size] = checkpoint['model_state']['net.%s_%d' % (name,i)].cpu().numpy().ravel()
            else:
                tokens = name.split('.')
                name_fixed = '.'.join(tokens[:3] + tokens[4:])
                w[i, offset:offset+size] = checkpoint['model_state']['net.%s_%d' % (name_fixed,i)].cpu().numpy().ravel()
            offset += size


    w[1] = 0.25 * (w[0] + w[2]) + 0.5 * w[1]

    mean = np.mean(w, axis=0)
    u = w[2] - w[0]
    du = np.linalg.norm(u)

    v = w[1] - w[0]
    v -= u / du * np.sum(u / du * v)
    dv = np.linalg.norm(v)

    u /= math.sqrt(3.0)
    v /= 1.5

    cov_factor = np.vstack((u[None, :], v[None, :]))
    subspace = SubspaceModel(torch.FloatTensor(mean), torch.FloatTensor(cov_factor))
    coords = np.dot(cov_factor / np.sum(np.square(cov_factor), axis=1, keepdims=True), (w - mean[None, :]).T).T
    theta = torch.FloatTensor(coords[2, :])

    for i in range(3):
        v = subspace(torch.FloatTensor(coords[i]))
        offset = 0
        for param in model.parameters():
            param.data.copy_(v[offset:offset + param.numel()].view(param.size()).to(args.device))
            offset += param.numel()
        utils.bn_update(loaders_bn['train'], model)
        print("Performance of model", i, "on the curve", end=":")
        print(utils.eval(loaders['test'], model, losses.cross_entropy))

else:
    assert len(args.checkpoint) == 1
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

    print('Loading: %s' % args.checkpoint[0])
    ckpt = torch.load(args.checkpoint[0])
    swag_model.load_state_dict(ckpt['state_dict'], strict=False)

    # first take as input SWA
    swag_model.set_swa()
    utils.bn_update(loaders_bn['train'], swag_model)
    print(utils.eval(loaders['test'], swag_model, losses.cross_entropy))

    mean, variance, cov_factor = swag_model.get_space()

    print(np.linalg.norm(cov_factor, axis=1))
    if args.random:
        cov_factor = cov_factor.cpu().numpy()
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

    subspace = SubspaceModel(mean, cov_factor)

    theta = torch.zeros(args.rank)


def log_pdf(theta, subspace, model, loader, criterion, temperature, device):
    w = subspace(torch.FloatTensor(theta))
    offset = 0
    for param in model.parameters():
        param.data.copy_(w[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()
    model.train()
    with torch.no_grad():
        loss = 0
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            batch_loss, _, _ = criterion(model, data, target)
            loss += batch_loss * data.size()[0]
    return -loss.item() / temperature


def oracle(theta):
    return log_pdf(
        theta,
        subspace=subspace,
        model=model,
        loader=loaders['train'],
        criterion=losses.cross_entropy,
        temperature=args.temperature,
        device=args.device
    )

ens_predictions = np.zeros((len(loaders['test'].dataset), num_classes))
targets = np.zeros((len(loaders['test'].dataset), num_classes))

columns = ['iter', 'log_prob', 'acc', 'nll', 'time']

samples = np.zeros((args.num_samples, args.rank))

for i in range(args.num_samples):
    time_sample = time.time()
    prior_sample = np.random.normal(loc=0.0, scale=args.prior_std, size=args.rank)
    theta, log_prob = elliptical_slice(initial_theta=theta.numpy().copy(), prior=prior_sample,
                                                    lnpdf=oracle)
    samples[i, :] = theta
    theta = torch.FloatTensor(theta)
    print(theta)
    w = subspace(theta)
    offset = 0
    for param in model.parameters():
        param.data.copy_(w[offset:offset + param.numel()].view(param.size()).to(args.device))
        offset += param.numel()

    utils.bn_update(loaders_bn['train'], model, subset=args.bn_subset)
    pred_res = utils.predict(loaders['test'], model)
    ens_predictions += pred_res['predictions']
    targets = pred_res['targets']
    time_sample = time.time() - time_sample
    values = ['%3d/%3d' % (i + 1, args.num_samples),
              log_prob,
              np.mean(np.argmax(ens_predictions, axis=1) == targets),
              nll(ens_predictions / (i + 1), targets),
              time_sample]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if i == 0:
        print(table)
    else:
        print(table.split('\n')[2])

ens_predictions /= args.num_samples
ens_acc = np.mean(np.argmax(ens_predictions, axis=1) == targets)
ens_nll = nll(ens_predictions, targets)
print("Ensemble NLL:", ens_nll)
print("Ensemble Accuracy:", ens_acc)

print("Ensemble Acc:", ens_acc)
print("Ensemble NLL:", ens_nll)

np.savez(
    os.path.join(args.dir, 'ens.npz'),
    seed=args.seed,
    samples=samples,
    ens_predictions=ens_predictions,
    targets=targets,
    ens_acc=ens_acc,
    ens_nll=ens_nll
)
