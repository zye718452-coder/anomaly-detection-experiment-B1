import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr = lr_ * (0.5 ** ((epoch - 1) // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_score2 = val_loss2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif val_loss > self.best_score - self.delta or val_loss2 > self.best_score2 - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.best_score2 = val_loss2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='train',
            dataset=self.dataset
        )
        self.vali_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='val',
            dataset=self.dataset
        )
        self.test_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='test',
            dataset=self.dataset
        )
        self.thre_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='thre',
            dataset=self.dataset
        )

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

        # ===== optional flags kept for compatibility =====
        self.use_ecr = bool(self.use_ecr) if hasattr(self, 'use_ecr') else False
        self.ecr_ma = self.ecr_ma if hasattr(self, 'ecr_ma') else 5
        self.ecr_eps = self.ecr_eps if hasattr(self, 'ecr_eps') else 1e-4
        self.lambda_ecr = self.lambda_ecr if hasattr(self, 'lambda_ecr') else 0.001

        self.use_ecrank = bool(self.use_ecrank) if hasattr(self, 'use_ecrank') else False
        self.rank_top_ratio = self.rank_top_ratio if hasattr(self, 'rank_top_ratio') else 0.25
        self.rank_margin = self.rank_margin if hasattr(self, 'rank_margin') else 0.05
        self.lambda_rank = self.lambda_rank if hasattr(self, 'lambda_rank') else 1e-4

        self.use_score_fusion = bool(self.use_score_fusion) if hasattr(self, 'use_score_fusion') else False
        self.score_fusion_ma = self.score_fusion_ma if hasattr(self, 'score_fusion_ma') else 5
        self.score_fusion_eps = self.score_fusion_eps if hasattr(self, 'score_fusion_eps') else 1e-4
        self.lambda_evi = self.lambda_evi if hasattr(self, 'lambda_evi') else 0.005

        # ===== dual-view configs =====
        self.use_dual_view = bool(self.use_dual_view) if hasattr(self, 'use_dual_view') else False
        self.dual_view_ma = self.dual_view_ma if hasattr(self, 'dual_view_ma') else 5
        self.dual_view_beta = self.dual_view_beta if hasattr(self, 'dual_view_beta') else 0.2
        self.dual_view_weight = self.dual_view_weight if hasattr(self, 'dual_view_weight') else 0.5

    def build_model(self):
        self.model = AnomalyTransformer(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
            e_layers=3
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def _moving_average(self, x, kernel_size):
        """
        x: [B, L, C]
        return: [B, L, C]
        """
        pad = (kernel_size - 1) // 2
        x_t = x.transpose(1, 2)  # [B, C, L]
        x_pad = F.pad(x_t, (pad, pad), mode='replicate')
        bg = F.avg_pool1d(x_pad, kernel_size=kernel_size, stride=1)
        return bg.transpose(1, 2)  # [B, L, C]

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                norm_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                    1, 1, 1, self.win_size
                )

                series_loss += (
                    torch.mean(my_kl_loss(series[u], norm_prior.detach())) +
                    torch.mean(my_kl_loss(norm_prior.detach(), series[u]))
                )

                prior_loss += (
                    torch.mean(my_kl_loss(norm_prior, series[u].detach())) +
                    torch.mean(my_kl_loss(series[u].detach(), norm_prior))
                )

            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)

            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    norm_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                        1, 1, 1, self.win_size
                    )

                    series_loss += (
                        torch.mean(my_kl_loss(series[u], norm_prior.detach())) +
                        torch.mean(my_kl_loss(norm_prior.detach(), series[u]))
                    )

                    prior_loss += (
                        torch.mean(my_kl_loss(norm_prior, series[u].detach())) +
                        torch.mean(my_kl_loss(series[u].detach(), norm_prior))
                    )

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1
                )
            )

            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(model_ref, 'last_alpha') and model_ref.last_alpha is not None:
                alpha = model_ref.last_alpha.detach().cpu().numpy()
                print(
                    "alpha mean: {:.6f}, std: {:.6f}, min: {:.6f}, max: {:.6f}".format(
                        alpha.mean(), alpha.std(), alpha.min(), alpha.max()
                    )
                )

            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def _compute_cri_from_output(self, input, output, series, prior, criterion, temperature=50):
        """
        Keep original anomaly score logic for one forward pass.
        """
        loss = torch.mean(criterion(input, output), dim=-1)
        series_loss = 0.0
        prior_loss = 0.0

        for u in range(len(prior)):
            norm_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                1, 1, 1, self.win_size
            )

            if u == 0:
                series_loss = my_kl_loss(series[u], norm_prior.detach()) * temperature
                prior_loss = my_kl_loss(norm_prior, series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], norm_prior.detach()) * temperature
                prior_loss += my_kl_loss(norm_prior, series[u].detach()) * temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        return cri

    def _make_second_view(self, input):
        """
        input: [B, L, C]
        second view = (1-beta) * input + beta * MA(input)
        """
        bg = self._moving_average(input, self.dual_view_ma)
        view2 = (1.0 - self.dual_view_beta) * input + self.dual_view_beta * bg
        return view2

    def _compute_dual_view_cri(self, input, criterion, temperature=50):
        """
        input: [B, L, C]
        return: cri [B, L]
        """
        # original view
        output1, series1, prior1, _ = self.model(input)
        cri1 = self._compute_cri_from_output(
            input=input,
            output=output1,
            series=series1,
            prior=prior1,
            criterion=criterion,
            temperature=temperature
        )

        if not self.use_dual_view:
            return cri1

        # second view
        input2 = self._make_second_view(input)
        output2, series2, prior2, _ = self.model(input2)
        cri2 = self._compute_cri_from_output(
            input=input2,
            output=output2,
            series=series2,
            prior=prior2,
            criterion=criterion,
            temperature=temperature
        )

        w = self.dual_view_weight
        cri = w * cri1 + (1.0 - w) * cri2
        return cri

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')
            )
        )
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")
        if self.use_dual_view:
            print(
                "Dual View ON | ma={} beta={} weight={}".format(
                    self.dual_view_ma, self.dual_view_beta, self.dual_view_weight
                )
            )
        else:
            print("Dual View OFF")

        criterion = nn.MSELoss(reduction='none')

        # (1) statistic on the train set
        attens_energy = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.train_loader):
                input = input_data.float().to(self.device)

                cri = self._compute_dual_view_cri(
                    input=input,
                    criterion=criterion,
                    temperature=temperature
                )

                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)

                cri = self._compute_dual_view_cri(
                    input=input,
                    criterion=criterion,
                    temperature=temperature
                )

                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)

                cri = self._compute_dual_view_cri(
                    input=input,
                    criterion=criterion,
                    temperature=temperature
                )

                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False

            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average='binary'
        )

        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score
            )
        )

        return accuracy, precision, recall, f_score