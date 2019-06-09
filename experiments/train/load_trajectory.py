import argparse
import torch

from swag import models
from swag.posteriors import SWAG

parser = argparse.ArgumentParser(description='Fit SWAG to SGD trajectory')

parser.add_argument('--path', type=str, default=None, required=True, help='target checkpoit file (default: None)')

parser.add_argument('--checkpoint', action='append')
parser.add_argument('--num_classes', type=int, default='10',
                    help='number of classes (default: 10)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--max_num_models', type=int, default=20,
                    help='maximum number of SWAG models to save')
parser.add_argument('--subspace', choices=['covariance', 'pca', 'freq_dir'], default='covariance')

args = parser.parse_args()

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Preparing model')
print(*model_cfg.args)
model = model_cfg.base(*model_cfg.args, num_classes=args.num_classes, **model_cfg.kwargs)
model.to(args.device)

swag_model = SWAG(model_cfg.base,
                  subspace_type=args.subspace,
                  subspace_kwargs={'max_rank': args.max_num_models},
                  *model_cfg.args, num_classes=args.num_classes, **model_cfg.kwargs)
swag_model.to(args.device)

for path in args.checkpoint:
    print(path)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])
    swag_model.collect_model(model)

torch.save({'state_dict': swag_model.state_dict()}, args.path)
