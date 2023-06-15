from __future__ import absolute_import

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from reid.lib.normalize import Normalize


__all__ = ['ftvitb16', 'ftvitl16']


class FtViT(nn.Module):
    __factory = {
        'ftvitb16': torchvision.models.vit_b_16,
        'ftvitl16': torchvision.models.vit_l_16,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, num_triplet_features=0, n_splits=10, batch_size=128, adjustment=None):
        super(FtViT, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.adjustment = adjustment

        # Construct base (pretrained) ViT
        if depth not in FtViT.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = FtViT.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classespyt
            self.num_triplet_features = num_triplet_features

            self.l2norm = Normalize(2)

            self.out_planes = self.base.heads[0].in_features
            self.class_token = nn.Parameter(torch.zeros(1, 1, self.out_planes))

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(self.out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
                init.constant_(self.feat_bn.weight, 1)
                init.constant_(self.feat_bn.bias, 0)
            else:
                self.num_features = self.out_planes
            if self.dropout >= 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

            # feature-wise Adjustment
            self.bn_list = nn.ModuleList()
            for _ in range(self.n_splits):
                self.bn_list.append(nn.Sequential(
                    nn.Linear(self.out_planes, self.num_features),
                    nn.BatchNorm1d(self.num_features),
                    nn.ReLU(inplace=True)
                ))
            self.feat_wise_classifiers = nn.ModuleList()
            for _ in range(self.n_splits):
                fc = nn.Linear(self.num_features, self.num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.feat_wise_classifiers.append(fc)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, output_feature=None):
        for name, module in self.base._modules.items():
            x = module(x)
            if name == "conv_proj":
                x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3))
                x = F.interpolate(x, size=(196), mode='linear')
                x = x.permute(0, 2, 1)
                batch_class_token = self.class_token.expand(x.size(0), -1, -1)
                # [128, 197, 768]
                x = torch.cat([batch_class_token, x], dim=1)
            if name == "encoder":
                break

        if self.cut_at_pooling:
            return x

        # [128, 768, 196]
        x = x[:, 1:].permute(0, 2, 1)

        if output_feature == 'feature-wise':
            split_h = x.size(2) / self.n_splits
            splits_precs = []
            for i in range(self.n_splits):
                split_size = x[:, :, int(i * split_h): int((i + 1) * split_h)].size(-1)
                # [128, 768, 1]
                split_feat = F.avg_pool1d(
                    x[:, :, int(i * split_h): int((i + 1) * split_h)], split_size)
                # [128, 768]
                split_feat = split_feat.view(x.size(0), -1)

                split_feat = self.bn_list[i](split_feat)

                # feature-wise Adjustment
                if output_feature == 'feature-wise':
                    split_feat = self.drop(split_feat)
                    prec = self.feat_wise_classifiers[i](split_feat)
                    splits_precs.append(prec)
            return splits_precs

        if output_feature == 'pool5':
            x = F.normalize(x)
            return x

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
            tgt_feat = F.normalize(x)
            tgt_feat = self.drop(tgt_feat)
            if output_feature == 'tgt_feat':
                return tgt_feat
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def ftvitb16(**kwargs):
    return FtViT('ftvitb16', **kwargs)


def ftvitl16(**kwargs):
    return FtViT('ftvitl16', **kwargs)

