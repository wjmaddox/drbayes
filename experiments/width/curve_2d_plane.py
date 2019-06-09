import argparse
import torch
import math
import numpy as np
import tabulate
import time

from swag import data, models, utils, losses

parser = argparse.ArgumentParser(description='Curve plane')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='/scratch/datasets/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')

parser.add_argument('--checkpoint', action='append')
parser.add_argument('--save_path', type=str, default=None, required=True, help='path to npz results file')

parser.add_argument('--margin', type=float, default=0.5, metavar='D', help='margin (default: 0.5)')
parser.add_argument('--N', type=int, default=21, metavar='N', help='number of points on a grid (default: 31)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

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
    use_validation=not args.use_test
)

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)

criterion = losses.cross_entropy

assert len(args.checkpoint) == 3

num_parameters = sum([p.numel() for p in model.parameters()])
w = np.zeros((3, num_parameters))
for i in range(3):
    print('Loading: %s' % args.checkpoint[i])
    ckpt = torch.load(args.checkpoint[i])
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt['model_state'])

    offset = 0
    for param in model.parameters():
        size = param.numel()
        w[i, offset:offset+size] = param.data.cpu().numpy().ravel()
        offset += size

w[1] = 0.25 * (w[0] + w[2]) + 0.5 * w[1]

mean = np.mean(w, axis=0)
u = w[2] - w[0]
du = np.linalg.norm(u)

v = w[1] - w[0]
v -= u / du * np.sum(u / du * v)
dv = np.linalg.norm(v)

print(du, dv)
u /= math.sqrt(3.0)
v /= 1.5

cov_factor = np.vstack((u[None, :], v[None, :]))
coords = np.dot(cov_factor / np.sum(np.square(cov_factor), axis=1, keepdims=True), (w - mean[None, :]).T).T

xs = np.linspace(-1.0 - args.margin, 1.0 + args.margin, args.N)
ys = np.linspace(-1.0 - args.margin, 1.0 + args.margin, args.N)

train_acc = np.zeros((args.N, args.N))
train_loss = np.zeros((args.N, args.N))
test_acc = np.zeros((args.N, args.N))
test_loss = np.zeros((args.N, args.N))

columns = ['x', 'y', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        t_start = time.time()
        w = mean + x * u + y * v

        offset = 0
        for param in model.parameters():
            size = np.prod(param.size())
            param.data.copy_(param.new_tensor(w[offset:offset+size].reshape(param.size())))
            offset += size

        utils.bn_update(loaders['train'], model)
        train_res = utils.eval(loaders['train'], model, criterion)
        test_res = utils.eval(loaders['test'], model, criterion)

        train_acc[i, j] = train_res['accuracy']
        train_loss[i, j] = train_res['loss']
        test_acc[i, j] = test_res['accuracy']
        test_loss[i, j] = test_res['loss']

        t = time.time() - t_start
        values = [x, y, train_loss[i, j], train_acc[i, j], test_loss[i, j], test_acc[i, j], t]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        if j == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

np.savez(
    args.save_path,
    xs=xs,
    ys=ys,
    train_acc=train_acc,
    train_err=100.0 - train_acc,
    train_loss=train_loss,
    test_acc=test_acc,
    test_err=100.0 - test_acc,
    test_loss=test_loss,
    coords=coords,
    du=np.linalg.norm(u),
    dv=np.linalg.norm(v),
)

