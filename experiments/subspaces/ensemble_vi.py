import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import tqdm
import tabulate

from swag import data, losses, models, utils
from swag.posteriors import SWAG, KFACLaplace
from swag.posteriors.vi_model import VIModel, ELBO

parser = argparse.ArgumentParser(description='SGD/SWA training')

parser.add_argument('--dir', type=str, default=None, help='directory (default: None)')
parser.add_argument('--save_file', type=str, default=None, required=True, help='path to npz results file')
parser.add_argument('--log_fname', type=str, default=None, required=True, help='file name for logging')
parser.add_argument('--vi_checkpoint', type=str, default=None, required=True, help='checkpoint')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='/scratch/datasets/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')
parser.add_argument('--N', type=int, default=10)
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--bn_subset', type=float, default=1.0, 
                    help='fraction of data to use for bn update')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.mean(np.log(ps + 1e-12))
    return nll

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
    split_classes=args.split_classes,
    shuffle_train=False
)
    
print('Loading: %s' % args.vi_checkpoint)
ckpt = torch.load(args.vi_checkpoint)

print('Preparing model')
vi_model = VIModel(
    mean=ckpt['state_dict']['mean'],
    cov_factor=ckpt['state_dict']['cov_factor'],
    num_classes=num_classes,
    base=model_cfg.base,
    *model_cfg.args,
    **model_cfg.kwargs
)

vi_model.load_state_dict(ckpt['state_dict'], strict=False)
vi_model = vi_model.cuda()

predictions = np.zeros((len(loaders['test'].dataset), num_classes))
targets = np.zeros(len(loaders['test'].dataset))

printf, logfile = utils.get_logging_print(os.path.join(args.dir, args.log_fname + '-%s.txt'))
print('Saving logs to: %s' % logfile)
columns = ['iter ens', 'acc', 'nll']

for i in range(args.N):
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

    values = ['%d/%d' % (i + 1, args.N), np.mean(np.argmax(predictions, axis=1) == targets), nll(predictions / (i+1), targets)]
    if i == 0:
        printf(tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f'))
    else:
        printf(tabulate.tabulate([values], columns, tablefmt='plain', floatfmt='8.4f').split('\n')[1])

predictions /= args.N
np.savez(os.path.join(args.dir, args.save_file), predictions=predictions, targets=targets)
