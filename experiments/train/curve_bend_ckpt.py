import argparse

import torch

from swag.posteriors import SWAG
from swag import data, losses, models, utils

parser = argparse.ArgumentParser(description='SGD/SWA training')

parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')


parser.add_argument('--ckpt', type=str, required=True, default=None, metavar='CKPT',
                    help='checkpoint to load (default: None)')

parser.add_argument('--save_path', type=str, default=None, required=True, help='path to save SWAG')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

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


model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)

print('Loading checkpoint %s' % args.ckpt)
checkpoint = torch.load(args.ckpt)

num_parameters = sum([p.numel() for p in model.parameters()])

offset = 0
for name, param in model.named_parameters():
    if 'net.%s_1' % name in checkpoint['model_state']:
        param.data.copy_(checkpoint['model_state']['net.%s_1' % name])
    else:
        # PRERESNET 164 fix
        tokens = name.split('.')
        name_fixed = '.'.join(tokens[:3] + tokens[4:])
        param.data.copy_(checkpoint['model_state']['net.%s_1' % name_fixed])


utils.bn_update(loaders['train'], model, verbose=True)

print(utils.eval(loaders['test'], model, losses.cross_entropy))

torch.save(
    {'state_dict': model.state_dict()},
    args.save_path,
)