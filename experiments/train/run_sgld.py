import argparse
import os, sys
import time
import tabulate

import torch
import numpy as np

from swag import data, models, utils, losses
from swag.posteriors import SWAG, SGLD

parser = argparse.ArgumentParser(description='SGD/SWA training')
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

parser.add_argument('--ckpt', type=str, default=None, required=True, metavar='CKPT',
                    help='pretrained checkpoint (default: None)')

parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of SGLD ensemble epochs (default: 160)')
parser.add_argument('--ens_start', type=int, default=11, metavar='N', help='ens start epoch (default: 11)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--lr_init', type=float, default=0.01, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--lr_final', type=float, default=0.0001, metavar='LR', help='final learning rate (default: 0.0001)')
parser.add_argument('--noise_factor', type=float, default=2e-5, metavar='N', help='SGLD noise term factor (default: 2e-5)')
parser.add_argument('--lr_gamma', type=float, default=0.55, metavar='LR', help='learning rate gamma (default: 0.55)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

parser.add_argument('--swag', action='store_true')
parser.add_argument('--swag_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--subspace', choices=['covariance', 'pca', 'freq_dir'], default='pca')
parser.add_argument('--max_num_models', type=int, default=20, help='maximum number of SWAG models to save')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')


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
model.to(args.device)


if args.swag:
    print('SGLD+SWAG training')
    swag_model = SWAG(model_cfg.base,
                      subspace_type=args.subspace, subspace_kwargs={'max_rank': args.max_num_models},
                      *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    swag_model.to(args.device)
else:
    print('SGLD training')


criterion = losses.cross_entropy

sgld_optimizer = SGLD(
    model.parameters(),
    lr=args.lr_init,
    weight_decay=args.wd,
    noise_factor=args.noise_factor
)

num_batches = len(loaders['train'])
num_iters = num_batches * (args.epochs - args.ens_start + 1)

lr_ratio = args.lr_init / args.lr_final
lr_b = (num_iters - 1) / (lr_ratio ** (1.0 / args.lr_gamma) - 1.0)
lr_a = args.lr_init * lr_b ** args.lr_gamma


def schedule(t):
    return lr_a / (lr_b + t) ** args.lr_gamma

lr = args.lr_init
print('Decaying LR from %g to %g with a=%g b=%g gamma=%g' %
      (schedule(0), schedule(num_iters - 1), lr_a, lr_b, args.lr_gamma))
print('Half-life LR: %g' % (schedule(num_iters / 2)))

print('Loading checkpoint %s' % args.ckpt)
checkpoint = torch.load(args.ckpt)
model.load_state_dict(checkpoint['state_dict'])

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'ens_loss', 'ens_acc', 'time', 'mem_usage']
if args.swag:
    columns = columns[:-2] + ['swa_te_loss', 'swa_te_acc'] + columns[-2:]
    swag_res = {'loss': None, 'accuracy': None}


sgld_ens_pred = None
sgld_targets = None
n_ensembled = 0.

ens_loss = None
ens_acc = None

n_iter = 0

for epoch in range(args.epochs):
    time_ep = time.time()

    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loaders['train']):
        if args.device.type == 'cuda':
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        loss, output, stats = criterion(model, input, target)

        lr = schedule(n_iter)
        utils.adjust_learning_rate(sgld_optimizer, lr)

        sgld_optimizer.zero_grad()
        loss.backward()
        sgld_optimizer.step()

        loss_sum += loss.data.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

        if epoch + 1 >= args.ens_start:
            n_iter += 1

    train_res = {
        'loss': loss_sum / len(loaders['train'].dataset),
        'accuracy': correct / len(loaders['train'].dataset) * 100.0,
    }

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None}

    if epoch + 1 >= args.ens_start:
        utils.bn_update(loaders['train'], model)
        out = utils.predict(loaders['test'], model)
        cur_pred = out['predictions']
        cur_targets = out['targets']

        if sgld_ens_pred is None:
            sgld_ens_pred = cur_pred.copy()
            sgld_targets = cur_targets.copy()
        else:
            sgld_ens_pred += (cur_pred - sgld_ens_pred) / (n_ensembled + 1)
        n_ensembled += 1

        idx = np.arange(sgld_targets.size)
        ens_loss = np.mean(-np.log(sgld_ens_pred[idx, sgld_targets]))
        ens_acc = np.mean(np.argmax(sgld_ens_pred, axis=-1) == sgld_targets) * 100.0

        np.savez(
            os.path.join(args.dir, "sgld_ens_preds.npz"), epoch=epoch, predictions=sgld_ens_pred, targets=sgld_targets)

    if args.swag and (epoch + 1) >= args.ens_start and (epoch + 1 - args.ens_start) % args.swag_c_epochs == 0:
        swag_model.collect_model(model)
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            swag_model.set_swa()
            utils.bn_update(loaders['train'], swag_model)
            swag_res = utils.eval(loaders['test'], swag_model, criterion)
        else:
            swag_res = {'loss': None, 'accuracy': None}

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
        )
        if args.swag and epoch + 1 >= args.ens_start:
            utils.save_checkpoint(
                args.dir,
                epoch + 1,
                name='swag',
                state_dict=swag_model.state_dict(),
            )

    time_ep = time.time() - time_ep
    memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)




    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'],
              ens_loss, ens_acc, time_ep, memory_usage]
    if args.swag:
        values = values[:-2] + [swag_res['loss'], swag_res['accuracy']] + values[-2:]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model.state_dict(),
    )
    if args.swag:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            name='swag',
            state_dict=swag_model.state_dict(),
        )

np.savez(
    os.path.join(args.dir, "sgld_ens_preds.npz"), epoch=args.epochs, predictions=sgld_ens_pred, targets=sgld_targets)
