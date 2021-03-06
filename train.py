"""Train YOLOv3 with random shapes."""
import argparse
import os
import logging
import time
import warnings
import numpy as np
import mxnet as mx
# from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from gluoncv import data as gdata
from gluoncv import utils as gutils
from model.model import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
# from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from utils.lr_scheduler import LRScheduler
from model.utils import LossMetricSimple, config, self_box_nms
from model.target import SelfDefaultTrainTransform
from mxnet import nd


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO networks with random input shape.')
    parser.add_argument('--network', type=str, default='darknet53',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=416,
                        help="Input data shape for evaluation, use 320, 416, 608... " +
                             "Training is with random shapes from (320 to 608).")
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=16, help='Number of data workers, you can use larger '
                                         'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./yolo3_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-mode', type=str, default='cosine',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='10,20',
                        help='epochs at which learning rate decays. default is 100,130.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=2,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Training images. Use -1 to automatically get the number.')
    parser.add_argument('--syncbn', action='store_true',
                        help='Use synchronize BN across devices.')
    parser.add_argument('--no-random-shape', action='store_true',
                        help='Use fixed size(data-shape) throughout the training, which will be faster '
                             'and require less memory. However, final model will be slightly worse.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether to enable mixup.')
    parser.add_argument('--no-mixup-epochs', type=int, default=10,
                        help='Disable mixup training if enabled in the last N epochs.')
    parser.add_argument('--fit-epoch', type=int, default=-1,
                        help='epoch at which open objectness probability fit. default -1, always close fit training')
    parser.add_argument('--label-smooth', action='store_true', help='Use label smoothing.')
    parser.add_argument('--coop-cfg', type=str, default='1',
                        help='coop configs. "," separate different output head, '
                             '" " separate different anchor and sig level in a same output layer. '
                             'such as 1,2 3 4,1 2 3')
    parser.add_argument('--margin', type=str, default='0')
    parser.add_argument('--thre-cls', type=str, default='1.25')
    parser.add_argument('--nms-mode', type=str, default='Default', choices=['Default', 'Exclude', 'Merge'])
    parser.add_argument('--coop-mode', type=str, default='flat', choices=['flat', 'convex', 'concave', 'equal'],
                        help='flat: different level grids have same weight loss in the training phase.'
                             'convex: center grids have higher weight than the marginal grids in the training phase.'
                             'concave: marginal grids have higher weight than the center grids in the training phase.'
                             'equal: consider the num of the same level grids to make loss equal')
    parser.add_argument('--sigma-weight', type=str, default='1.2',
                        help='when the coop_mode is convex or concave, they need a Gaussian sigma')
    parser.add_argument('--results-dir', default='result_test', help='path to save results')
    parser.add_argument('--pretrained', action='store_true', help='whether to train for detection checkpoint.')
    parser.add_argument('--ignore-iou-thresh', type=float, default=0.7)
    parser.add_argument('--specific-anchor', type=str, default='default',
                        choices=['default', 'rectangle', 'rectanglefix', 'square'])
    parser.add_argument('--sa-level', type=int, default=2, choices=[1, 2, 3])
    parser.add_argument('--sq-level', type=int, default=5)
    parser.add_argument('--coop-loss', action='store_true', help='whether to train with cooperation loss.')
    parser.add_argument('--separate', action='store_true', help='whether to train coord and obj separately.')
    args = parser.parse_args()
    return args


def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        root = os.path.join('~', 'dataset', 'coco2017')
        train_dataset = gdata.COCODetection(root=root, splits='instances_train2017', use_crowd=False)
        val_dataset = gdata.COCODetection(root=root, splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, os.path.join(args.results_dir, args.save_prefix + '_eval'), cleanup=True,
            data_shape=(args.data_shape, args.data_shape))
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if args.num_samples < 0:
        args.num_samples = len(train_dataset)
    if args.mixup:
        from gluoncv.data import MixupDetection
        train_dataset = MixupDetection(train_dataset)
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, args):
    """Get dataloader."""
    width, height = data_shape, data_shape
    end = 10 if args.coop_loss else 8
    batchify_fn = Tuple(*([Stack() for _ in range(end)] + [Pad(axis=0, pad_val=-1) for _ in
                                                           range(1)]))  # stack image, all targets generated
    if args.no_random_shape:
        train_loader = gluon.data.DataLoader(train_dataset.transform(SelfDefaultTrainTransform(width, height, net,
            mixup=args.mixup, coop_configs=args.coop_cfg, margin=args.margin, thre_cls=args.thre_cls,
            coop_loss=args.coop_loss, coop_mode=args.coop_mode, sigma_weight=args.sigma_weight, label_smooth=args.label_smooth)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    else:
        transform_fns = [SelfDefaultTrainTransform(x * 32, x * 32, net, coop_configs=args.coop_cfg,
            mixup=args.mixup, margin=args.margin, thre_cls=args.thre_cls, coop_loss=args.coop_loss,
            coop_mode=args.coop_mode, sigma_weight=args.sigma_weight, label_smooth=args.label_smooth) for x in range(10, 20)]
        train_loader = RandomTransformDataLoader(
            transform_fns, train_dataset, batch_size=batch_size, interval=10, last_batch='rollover',
            shuffle=True, batchify_fn=batchify_fn, num_workers=num_workers)
    val_batchify_fn = Tuple((Stack(), Pad(pad_val=-1)))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


def save_params(net, best_map, current_map, epoch, save_interval, prefix, result_dir):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters(os.path.join(result_dir, '{:s}_best.params'.format(prefix, epoch, current_map)))
        with open(os.path.join(result_dir, prefix + '_best_map.log'), 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        net.save_parameters(os.path.join(result_dir, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))


def validate(net, val_data, ctx, eval_metric, nms_mode):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            # to deal with the self nms and the default nms separately
            if nms_mode == 'Default':
                ids, scores, bboxes = net(x)
            # else:
            #     results = net(x)
            #     ids, scores, bboxes = self_box_nms(
            #         results, overlap_thresh=0.45, valid_thresh=0.01, topk=400, nms_style=nms_mode)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
        # if nms_mode != 'Default':
        #     mxim = 0
        #     for det_bbox in det_bboxes:
        #         if det_bbox.shape[1] > mxim:
        #             mxim = det_bbox.shape[1]
        #     for ind in range(len(det_bboxes)):
        #         det_bboxes[ind] = nd.pad(det_bboxes[ind], mode='constant', pad_width=(
        #             0, 0, 0, mxim - det_bboxes[ind].shape[1], 0, 0), constant_value=-1)
        #         gt_ids[ind] = nd.pad(gt_ids[ind], mode='constant', pad_width=(
        #             0, 0, 0, mxim - gt_ids[ind].shape[1], 0, 0), constant_value=-1)
        #         gt_bboxes[ind] = nd.pad(gt_bboxes[ind], mode='constant', pad_width=(
        #             0, 0, 0, mxim - gt_bboxes[ind].shape[1], 0, 0), constant_value=-1)
        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()


def train(net, train_data, val_data, eval_metric, ctx, args, logger):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    if args.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0
    if args.lr_decay_period > 0:
        lr_decay_epoch = list(range(args.lr_decay_period, args.epochs, args.lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
    lr_scheduler = LRScheduler(mode=args.lr_mode,
                               baselr=args.lr,
                               niters=args.num_samples // args.batch_size,
                               nepochs=args.epochs,
                               step=lr_decay_epoch,
                               step_factor=args.lr_decay, power=2,
                               warmup_epochs=args.warmup_epochs)

    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'wd': args.wd, 'momentum': args.momentum, 'lr_scheduler': lr_scheduler},
        kvstore='local')

    metric_loss = LossMetricSimple(args.coop_loss)

    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        if args.mixup:
            # TODO(zhreshold): more elegant way to control mixup during runtime
            try:
                train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
            except AttributeError:
                train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
            if epoch >= args.epochs - args.no_mixup_epochs:
                try:
                    train_data._dataset.set_mixup(None)
                except AttributeError:
                    train_data._dataset._data.set_mixup(None)

        # TODO: more elegant way to control fit during runtime
        if (args.fit_epoch > 0) and (epoch >= args.fit_epoch):
            for fns in train_data._transform_fns:
                fns.set_prob_fit(True)

        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        net.hybridize()
        test = 0
        for i, batch in enumerate(train_data):
            test += 1
            if test > 200:
                break
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            # objectness, center_targets, scale_targets, weights, class_mask, obj_mask
            end = 10 if args.coop_loss else 8
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, end)]
            gt_boxes = gluon.utils.split_and_load(batch[end], ctx_list=ctx, batch_axis=0)
            sum_losses = []
            metric_loss.initial()
            with autograd.record():
                for ix, x in enumerate(data):
                    loss_list = net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                    # a = loss_list[0:4] + loss_list[5:]
                    sum_losses.append(sum([l for l in loss_list]))
                    metric_loss.append(loss_list)
                autograd.backward(sum_losses)
            lr_scheduler.update(i, epoch)
            trainer.step(batch_size)
            metric_loss.update()
            if args.log_interval and not (i + 1) % args.log_interval:
                name_loss_str, name_loss = metric_loss.get()
                logger.info(('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, ' + name_loss_str).format(
                    epoch, i, trainer.learning_rate, batch_size / (time.time() - btic), *name_loss))
            btic = time.time()

        name_loss_str, name_loss = metric_loss.get()
        logger.info(('[Epoch {}] Training cost: {:.3f}, ' + name_loss_str).format(epoch, (time.time() - tic), *name_loss))
        if not (epoch + 1) % args.val_interval:
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric, args.nms_mode)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, args.save_interval, args.save_prefix, args.results_dir)


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    fh = logging.FileHandler(os.path.join(args.results_dir, log_file_path))
    logger.addHandler(fh)
    logger.info(args)
    config(args)

    # network
    net_name = '_'.join(('yolo3', args.network, args.dataset))
    # args.save_prefix += net_name
    # use sync bn if specified
    if args.syncbn and len(ctx) > 1:
        net = get_model(net_name, pretrained=args.pretrained, norm_layer=gluon.contrib.nn.SyncBatchNorm,
                        norm_kwargs={'num_devices': len(ctx)}, coop_configs=args.coop_cfg, nms_mode=args.nms_mode,
                        ignore_iou_thresh=args.ignore_iou_thresh, specific_anchor=args.specific_anchor,
                        sa_level=args.sa_level, sq_level=args.sq_level, coop_loss=args.coop_loss)
        # used by cpu worker
        async_net = get_model(net_name, pretrained_base=False, coop_configs=args.coop_cfg, specific_anchor=args.specific_anchor,
                              sa_level=args.sa_level, sq_level=args.sq_level, coop_loss=args.coop_loss)
    else:
        net = get_model(net_name, pretrained=args.pretrained, coop_configs=args.coop_cfg, nms_mode=args.nms_mode,
                        ignore_iou_thresh=args.ignore_iou_thresh, specific_anchor=args.specific_anchor,
                        sa_level=args.sa_level, sq_level=args.sq_level, coop_loss=args.coop_loss)
        async_net = net
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
        async_net.load_parameters(args.resume.strip())
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
    train_data, val_data = get_dataloader(
        async_net, train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers, args)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args, logger)
