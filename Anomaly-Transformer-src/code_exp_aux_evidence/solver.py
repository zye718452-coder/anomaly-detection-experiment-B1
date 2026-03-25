import os
import time
import numpy as np
import torch
import torch.nn as nn

from torch import optim

from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment


def my_kl_loss(p, q):
    """
    p, q: [B, H, L, L]
    """
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class EarlyStopping:
    def __init__(self, patience=3, verbose=False, dataset_name='', delta=0.0):
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
        score = -val_loss
        score2 = -val_loss2

        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_path = os.path.join(path, str(self.dataset) + '_checkpoint.pth')
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.device = torch.device('cuda:%d' % self.gpu if self.use_gpu else 'cpu')

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

    def build_model(self):
        self.model = AnomalyTransformer(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            output_attention=self.output_attention,
            aux_evidence=bool(self.aux_evidence),
            aux_evidence_weight=self.aux_evidence_weight,
            aux_ma_kernel=self.aux_ma_kernel
        ).float().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if self.pretrained_model is not None:
            pretrained_str = str(self.pretrained_model)
            if pretrained_str.endswith('.pth') and os.path.exists(pretrained_str):
                print('Loading pretrained model:', pretrained_str)
                self.model.load_state_dict(torch.load(pretrained_str, map_location=self.device))
            else:
                print('Skip loading pretrained_model:', pretrained_str)

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []

        criterion = nn.MSELoss()

        with torch.no_grad():
            for i, (input_data, _) in enumerate(vali_loader):
                input_data = input_data.float().to(self.device)

                output, series, prior, sigmas, aux = self.model(input_data)

                rec_loss = criterion(output, input_data)

                evidence_loss = 0.0
                if self.aux_evidence:
                    evidence_loss = criterion(aux["pred_evidence"], aux["target_evidence"])

                total_rec_loss = rec_loss + self.aux_evidence_weight * evidence_loss if self.aux_evidence else rec_loss

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

                loss1 = total_rec_loss - self.k * series_loss
                loss2 = total_rec_loss + self.k * prior_loss

                loss_1.append(loss1.item())
                loss_2.append(loss2.item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)
        criterion = nn.MSELoss()

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()

            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1

                input_data = input_data.float().to(self.device)

                output, series, prior, sigmas, aux = self.model(input_data)

                rec_loss = criterion(output, input_data)

                evidence_loss = 0.0
                if self.aux_evidence:
                    evidence_loss = criterion(aux["pred_evidence"], aux["target_evidence"])

                total_rec_loss = rec_loss + self.aux_evidence_weight * evidence_loss if self.aux_evidence else rec_loss

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

                loss1 = total_rec_loss - self.k * series_loss
                loss2 = total_rec_loss + self.k * prior_loss

                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

                loss1_list.append(loss1.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)
            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss1: {3:.7f} Vali Loss2: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss1, vali_loss2
                )
            )

            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(self.model_save_path, str(self.dataset) + '_checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

    def test(self):
        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')
        best_model_path = os.path.join(self.model_save_path, str(self.dataset) + '_checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.eval()

        # (1) statistic on train set
        attens_energy = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.train_loader):
                input_data = input_data.float().to(self.device)

                output, series, prior, sigmas, _ = self.model(input_data)

                loss = torch.mean(criterion(input_data, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    norm_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                        1, 1, 1, self.win_size
                    )

                    if u == 0:
                        series_loss = my_kl_loss(series[u], norm_prior.detach()) * self.k
                        prior_loss = my_kl_loss(norm_prior, series[u].detach()) * self.k
                    else:
                        series_loss += my_kl_loss(series[u], norm_prior.detach()) * self.k
                        prior_loss += my_kl_loss(norm_prior, series[u].detach()) * self.k

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)

        # (2) statistic on threshold set
        test_energy = []
        test_labels = []

        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input_data = input_data.float().to(self.device)

                output, series, prior, sigmas, _ = self.model(input_data)

                loss = torch.mean(criterion(input_data, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    norm_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                        1, 1, 1, self.win_size
                    )

                    if u == 0:
                        series_loss = my_kl_loss(series[u], norm_prior.detach()) * self.k
                        prior_loss = my_kl_loss(norm_prior, series[u].detach()) * self.k
                    else:
                        series_loss += my_kl_loss(series[u], norm_prior.detach()) * self.k
                        prior_loss += my_kl_loss(norm_prior, series[u].detach()) * self.k

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()

                test_energy.append(cri)
                test_labels.append(labels)

        test_energy = np.concatenate(test_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)

        combined_energy = np.concatenate([attens_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on test set
        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

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