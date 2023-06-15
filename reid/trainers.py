from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
from tqdm import  tqdm

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
import copy
import numpy as np
import visdom
import os
import torch.nn.functional as F


class ExemplarMemory(Function):
    # def __init__(self, em, alpha=0.01):
    #     super(ExemplarMemory, self).__init__()
    #     self.em = em
    #     self.alpha = alpha

    @staticmethod
    def forward(ctx, inputs, targets, em, alpha=0.01):
        ctx.save_for_backward(inputs, targets)
        ctx.em = em
        ctx.alpha = alpha
        outputs = inputs.mm(em.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.em)
        for x, y in zip(inputs, targets):
            ctx.em[y] = ctx.alpha * ctx.em[y] + (1. - ctx.alpha) * x
            ctx.em[y] /= ctx.em[y].norm()
        return grad_inputs, None, None, None


class Trainer(object):
    def __init__(self, model, lmd=0.5, n_splits=8, adjustment='feature-wise', num_classes=0, num_features=0, images_len=0):
        super(Trainer, self).__init__()
        self.n_splits = n_splits
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.pid_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.lmd = lmd
        self.adjustment = adjustment
        self.num_classes = num_classes
        self.num_features = num_features
        self.mean_feat = None
        self.images_len = images_len

        # Exemplar memory
        self.em = nn.Parameter(torch.zeros(num_classes, num_features))
        self.km = nn.Parameter(torch.zeros(images_len))

    def get_all_mean_feat(self, data_loader):
        with torch.no_grad():
            self.all_mean_feat = torch.zeros(self.n_splits, self.num_classes, self.num_features).cuda()
            count = torch.zeros(self.num_classes).cuda()
            pbar = tqdm(data_loader)
            for i, inputs in enumerate(pbar):
                inputs, pids = self._parse_data(inputs)
                # feats[n_splits, 128, 2048]
                feats = self.model(inputs, 'mean_feat')
                for j in range(feats.size(0)):
                    for k in range(feats.size(1)):
                        if j == 0:
                            count[pids[k]] = count[pids[k]] + 1
                        self.all_mean_feat[j, pids[k]] = self.all_mean_feat[j, pids[k]] + feats[j, k]

                pbar.set_description("Get All Mean Feat")
            self.all_mean_feat = self.all_mean_feat / count.view(1, -1, 1)
        return self.all_mean_feat

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.set_model_train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        mean_feat = None
        if self.adjustment == 'class-wise' or self.adjustment == 'Combined':
            mean_feat = self.get_all_mean_feat(data_loader)

        # Train
        pbar = tqdm(data_loader)
        for i, inputs in enumerate(pbar):
            data_time.update(time.time() - end)

            # inputs
            inputs, pids = self._parse_data(inputs)

            outputs = None
            if self.adjustment == 'feature-wise':
                outputs = self.model(inputs, self.adjustment)
            elif self.adjustment == 'class-wise' and epoch > 0:
                outputs = self.model(inputs, self.adjustment, mean_feat=mean_feat)
            elif self.adjustment == 'Combined' and epoch == 0:
                outputs = self.model(inputs, self.adjustment)
            elif self.adjustment == 'Combined' and epoch > 0:
                outputs = self.model(inputs, self.adjustment, mean_feats=mean_feat)
            else:
                outputs = self.model(inputs)
            loss = 0
            prec1 = 0
            if self.adjustment == 'feature-wise' or self.adjustment == 'Combined':
                for j in range(self.n_splits):
                    loss_item = self.pid_criterion(outputs[j], pids)
                    loss = loss + loss_item
                loss /= self.n_splits
                sum_prec = torch.sum(F.softmax(torch.stack(outputs), dim=2), dim=0) / self.n_splits
                prec, = accuracy(sum_prec.data, pids.data)
                prec1 = prec[0]
            else:
                loss = self.pid_criterion(outputs, pids)
                prec, = accuracy(outputs.data, pids.data)
                prec1 = prec[0]

            loss_print = {}
            loss_print['loss'] = loss.item()

            losses.update(loss.item(), inputs.size(0))
            precisions.update(prec1, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                log = "Epoch: [{}], Time {:.3f}, Loss {:.3f}, Prec {:.2%}" \
                    .format(epoch,
                            batch_time.sum,
                            losses.val,
                            precisions.val)

                for tag, value in loss_print.items():
                    log += ", {}: {:.4f}".format(tag, value)
                pbar.set_description("%s"%(log))

        log = "Epoch: [{}], Time {:.3f} ({:.3f}), Data {:.3f} ({:.3f}), Loss {:.3f} ({:.3f}), Prec {:.2%} ({:.2%})" \
            .format(epoch,
                    batch_time.sum, batch_time.avg,
                    data_time.sum, data_time.avg,
                    losses.val, losses.avg,
                    precisions.val, precisions.avg)
        for tag, value in loss_print.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        return inputs, pids

    def set_model_train(self):
        self.model.train()

        # Fix first BN
        fixed_bns = []
        for idx, (name, module) in enumerate(self.model.module.named_modules()):
            if name.find("layer3") != -1:
                # assert len(fixed_bns) == 22
                break
            if name.find("bn") != -1:
                fixed_bns.append(name)
                module.eval()

    def smooth_hot(self, inputs, targets, k=6):
        # Sort
        _, index_sorted = torch.sort(inputs, dim=1, descending=True)
        m, n = index_sorted.size()

        ones_mat = torch.ones(targets.size(0), k).to(self.device)
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.zeros(inputs.size()).to(self.device)
        for i in range(m):
            k_reciprocal_neigh = self.get_k_reciprocal_neigh(index_sorted, i, k)
            weights = 3.0/k
            targets_onehot[i, k_reciprocal_neigh] = weights

        targets_onehot.scatter_(1, targets, float(1))
        return targets_onehot

    def get_k_reciprocal_neigh(self, index_sorted, i, k):
        k_neigh_idx = index_sorted[i, :k]
        neigh_sims = self.em[k_neigh_idx].mm(self.em.t())

        _, neigh_idx_sorted = torch.sort(neigh_sims.detach().clone(), dim=1, descending=True)
        fi = np.where(neigh_idx_sorted[:, :k].cpu() == index_sorted.cpu()[i, 0])[0]
        return index_sorted[i, fi]
