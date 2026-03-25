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

    # auxiliary evidence task
    parser.add_argument('--aux_evidence', type=int, default=1,
                        help='1: enable auxiliary evidence reconstruction, 0: disable')
    parser.add_argument('--aux_evidence_weight', type=float, default=0.03,
                        help='weight of auxiliary evidence loss')
    parser.add_argument('--aux_ma_kernel', type=int, default=5,
                        help='moving average kernel for evidence target')

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