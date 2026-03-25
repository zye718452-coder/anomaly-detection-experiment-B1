import argparse
import os
import torch
from solver import Solver


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(config):
    cudnn = True
    if cudnn and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
        solver.test()
    elif config.mode == 'test':
        solver.test()
    else:
        raise ValueError("mode must be 'train' or 'test'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=float, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--data_path', type=str, default='dataset/SMD')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')

    # model config
    parser.add_argument('--anormly_ratio', type=float, default=0.5)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', type=str2bool, default=True)

    # device
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    # Eq.11: pure post-hoc score fusion
    parser.add_argument('--use_score_fusion', type=int, default=0)
    parser.add_argument('--lambda_evi', type=float, default=0.005)
    parser.add_argument('--score_fusion_ma', type=int, default=5)
    parser.add_argument('--score_fusion_eps', type=float, default=1e-4)

    config = parser.parse_args()

    if config.use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)

    print('------------ Options -------------')
    for k, v in sorted(vars(config).items()):
        print(f'{k}: {v}')
    print('-------------- End ----------------')

    main(config)