import os
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)

    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)

    # =========================
    # score fusion arguments
    # =========================
    parser.add_argument('--use_score_fusion', type=int, default=0,
                        help='whether to use evidence-guided score fusion at test time')
    parser.add_argument('--score_fusion_ma', type=int, default=5,
                        help='moving average window for evidence extraction')
    parser.add_argument('--score_fusion_eps', type=float, default=1e-4,
                        help='epsilon for temporal standardization in score fusion')
    parser.add_argument('--lambda_evi', type=float, default=0.005,
                        help='fusion weight for evidence score')

    # =========================
    # ECR arguments
    # =========================
    parser.add_argument('--use_ecr', type=int, default=0,
                        help='whether to use evidence-consistency regularization')
    parser.add_argument('--ecr_ma', type=int, default=5,
                        help='moving average window for channel evidence extraction')
    parser.add_argument('--ecr_eps', type=float, default=1e-4,
                        help='epsilon for normalization in ECR')
    parser.add_argument('--lambda_ecr', type=float, default=0.001,
                        help='weight of evidence-consistency regularization')

    # =========================
    # ECRank arguments
    # =========================
    parser.add_argument('--use_ecrank', type=int, default=0,
                        help='whether to use evidence-guided channel ranking regularization')
    parser.add_argument('--rank_top_ratio', type=float, default=0.25,
                        help='top/bottom channel ratio used in ranking regularization')
    parser.add_argument('--rank_margin', type=float, default=0.05,
                        help='margin for ranking regularization')
    parser.add_argument('--lambda_rank', type=float, default=0.0001,
                        help='weight of ranking regularization')

    # =========================
    # dual-view inference arguments
    # =========================
    parser.add_argument('--use_dual_view', type=int, default=0,
                        help='whether to use dual-view inference ensemble at test time')
    parser.add_argument('--dual_view_ma', type=int, default=5,
                        help='moving average window for the second view')
    parser.add_argument('--dual_view_beta', type=float, default=0.2,
                        help='blend ratio for the smoothed second view')
    parser.add_argument('--dual_view_weight', type=float, default=0.5,
                        help='weight for original-view score in final ensemble')

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)