import os
import argparse
import torch
from torch.backends import cudnn

from solver import Solver


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(config):
    cudnn.benchmark = True

    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    else:
        raise ValueError("mode must be 'train' or 'test'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/SMD/')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=1.0)

    # model structure
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', type=str2bool, default=True)

    # GPU
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    # EGAP: evidence-guided association purification
    parser.add_argument('--evidence_bias', type=int, default=1,
                        help='1: enable evidence-guided attention bias, 0: disable')
    parser.add_argument('--evidence_lambda', type=float, default=0.1,
                        help='strength of evidence attention bias')
    parser.add_argument('--evidence_ma_kernel', type=int, default=5,
                        help='moving average kernel for raw-input background')
    parser.add_argument('--evidence_local_kernel', type=int, default=5,
                        help='local statistics kernel for evidence scorer')
    parser.add_argument('--evidence_hidden_dim', type=int, default=16,
                        help='hidden dim for evidence scorer conv')
    parser.add_argument('--evidence_share_across_heads', type=int, default=1,
                        help='1: share same evidence bias across heads')

    config = parser.parse_args()

    if config.use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
    else:
        config.use_gpu = False

    print('------------ Options -------------')
    for k, v in sorted(vars(config).items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)