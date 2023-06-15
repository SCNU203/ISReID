from __future__ import absolute_import

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from reid.lib.normalize import Normalize


__all__ = ['clvitb16', 'clvitl16']


class ClViT(nn.Module):
    __factory = {
        'clvitb16': torchvision.models.vit_b_16,
        'clvitl16': torchvision.models.vit_l_16,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, num_triplet_features=0, n_splits=10, batch_size=128, adjustment=None):
        super(ClViT, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.batch_size = batch_size
        self.adjustment = adjustment

        # Construct base (pretrained) ViT
        if depth not in ClViT.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ClViT.__factory[depth](pretrained=pretrained)

        self.class_token = nn.Parameter(torch.zeros(1, 1, 768))
        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            self.num_triplet_features = num_triplet_features

            self.l2norm = Normalize(2)

            self.out_planes = self.base.heads[0].in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(self.out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
                init.constant_(self.feat_bn.weight, 1)
                init.constant_(self.feat_bn.bias, 0)
            else:
                self.num_features = out_planes
            if self.dropout >= 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

            # class-wise Adjustment
            self.class_wise_classifier = nn.Linear(self.num_features * 2, self.num_classes)
            init.normal_(self.class_wise_classifier.weight, std=0.001)
            init.constant_(self.class_wise_classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, output_feature=None, mean_feat=None):
        for name, module in self.base._modules.items():
            x = module(x)
            if name == "conv_proj":
                x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3))
                x = F.interpolate(x, size=(196), mode='linear')
                x = x.permute(0, 2, 1)
                batch_class_token = self.class_token.expand(x.size(0), -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
            if name == "encoder":
                break

        if self.cut_at_pooling:
            return x

        x = x[:, 1:].permute(0, 2, 1)
        x = F.avg_pool1d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

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
        if output_feature == 'mean_feat':
            x = F.relu(x)
            return torch.stack([x])
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)

        # class-wise Adjustment
        if output_feature == 'class-wise':
            mean_feat = self.drop(mean_feat[0])
            pre_class = F.softmax(self.classifier(x), dim=1)
            cwa = pre_class.mm(mean_feat)
            output = self.class_wise_classifier(torch.cat([x, cwa], dim=1))
            return output
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


def clvitb16(**kwargs):
    return ClViT('clvitb16', **kwargs)


def clvitl16(**kwargs):
    return ClViT('clvitl16', **kwargs)

