from __future__ import print_function, absolute_import
import argparse
import datetime
import os
import os.path as osp
import time

import numpy as np
import sys
import torch
from tqdm import tqdm
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid.datasets.dataset_analysis import DA

from reid import models
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor, UnsupervisedCamStylePreprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


def get_data(data_dir, train_dataset, test_dataset, height, width, batch_size, re=0, workers=8):

    dataset = DA(data_dir, train_dataset, test_dataset)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(EPSILON=re),
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(dataset.train, root=osp.join(dataset.images_dir, dataset.train_path),
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.test_dir, dataset.query_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.test_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, query_loader, gallery_loader, len(dataset.train)


def main(args):
    if args.adjustment == 'feature-wise':
        args.arch = 'ft' + args.arch
    if args.adjustment == 'class-wise':
        args.arch = 'cl' + args.arch
        args.n_splits = 1
    if args.adjustment == 'Combined':
        args.arch = 'cb' + args.arch

    # For fast training.
    cudnn.benchmark = True
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print(torch.cuda.is_available())
    # cord = torch.randn(268435456 * args.core).to(device)

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print('log_dir=', args.logs_dir)

    # Print logs
    print(args)

    # Create data loaders
    dataset, num_classes, train_loader, \
    query_loader, gallery_loader, images_len = get_data(args.data_dir, args.dataset, args.test, args.height,
                                            args.width, args.batch_size,
                                            args.re, args.workers)

    # Create model
    if args.adjustment == 'none':
        model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes)
    else:
        model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes, n_splits=args.n_splits, batch_size=args.batch_size, adjustment=args.adjustment)


    # Load from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {} "
              .format(start_epoch))

    # Set model
    model = nn.DataParallel(model).to(device)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(query_loader, gallery_loader, dataset.query,
                           dataset.gallery, args.output_feature)
        return

    # Optimizer
    base_param_ids = set(map(id, model.module.base.parameters()))

    base_params_need_for_grad = filter(lambda p: p.requires_grad, model.module.base.parameters())

    new_params = [p for p in model.parameters() if
                    id(p) not in base_param_ids]
    param_groups = [
        {'params': base_params_need_for_grad, 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]

    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Trainer
    trainer = Trainer(model, n_splits=args.n_splits, adjustment=args.adjustment, num_classes=num_classes, num_features=args.features, images_len=images_len)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = args.epochs_decay
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer=optimizer)

        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
        }, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        # print('\n * Finished epoch {:3d} \n'.
        #       format(epoch))
        # if epoch % 10 == 0 and epoch != 0:
        # if epoch >= args.epochs - 1:
        #     print('Test in epoch', epoch, ':')
        #     evaluator = Evaluator(model)
        #     evaluator.evaluate(query_loader, gallery_loader, dataset.query,
        #                        dataset.gallery, args.output_feature)

    # Final test
    print('Test with best model:')
    evaluator = Evaluator(model)
    evaluator.evaluate(query_loader, gallery_loader, dataset.query,
                       dataset.gallery, args.output_feature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Invariance Learning for Domain Adaptive Re-ID")
    # dataet
    parser.add_argument('-d', '--dataset', type=str, default='duke',
                        choices=['market', 'duke', 'msmt17', 'cuhk03'])
    parser.add_argument('-t', '--test', type=str, default='duke',
                        choices=['market', 'duke', 'msmt17', 'cuhk03'])

    parser.add_argument('--gpus', type=str, help='gpus')
    # imgs setting
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=384,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for ImageNet pretrained"
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--epochs_decay', type=int, default=40)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    # random erasing
    parser.add_argument('--re', type=float, default=0.5)

    parser.add_argument('--n_splits', default=12, type=int,
                        help='把 feature 分成 n_split 层')
    parser.add_argument('--adjustment', default='feature-wise', type=str,
                        help='调整方式')

    args = parser.parse_args()
    main(args)

