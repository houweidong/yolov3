"""YOLO Demo script."""
import os
import argparse
import mxnet as mx
import gluoncv as gcv
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt
from model.model import get_model
from gluoncv.data import COCODetection
import numpy as np
from model.utils import get_coop_config


def parse_args():
    parser = argparse.ArgumentParser(description='Test with YOLO networks.')
    parser.add_argument('--network', type=str, default='yolo3_darknet53_coco',
                        help="Base network name")
    parser.add_argument('--images', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='true',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.0,
                        help='Threshold of object score when visualize the bboxes.')
    parser.add_argument('--classes', type=str, default='',
                        help='classes to be displayed. could more than one class')
    parser.add_argument('--coop-cfg', type=str, default='2, 2, 2',
                        help='coop configs. "," separate different output head, '
                             '" " separate different sig level in a same output layer. '
                             'such as 1,2 3 4,1 2 3')
    parser.add_argument('--nms-mode', type=str, default='Default', choices=['Default', 'Exclude', 'Merge'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    # grab some image if not specified
    if not args.images.strip():
        gcv.utils.download("https://cloud.githubusercontent.com/assets/3307514/" +
                           "20012568/cbc2d6f6-a27d-11e6-94c3-d35a9cb47609.jpg", 'street.jpg')
        image_list = ['street.jpg']
    else:
        image_list = [x.strip() for x in args.images.split(',') if x.strip()]

    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(args.network, pretrained=True)
    else:
        coop_cfg = get_coop_config(args.coop_cfg)
        net = get_model(args.network, pretrained_base=False, coop_configs=coop_cfg, nms_mode=args.nms_mode)
        net.load_parameters(args.pretrained)
    net.set_nms(0.000001, 200)
    net.collect_params().reset_ctx(ctx=ctx)

    for image in image_list:
        ax = None
        x, img = presets.yolo.load_test(image, short=512)
        x = x.as_in_context(ctx[0])
        ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
        # select args.classes in ids
        if args.classes:
            cla_ids = []
            clses = list(filter(None, args.classes.split(',')))
            cond_a = np.zeros_like(ids, dtype=np.int)
            for cls in clses:
                cla_ids.append(COCODetection.CLASSES.index(cls))
                for cid in cla_ids:
                    cond = cid == ids.astype(np.int)
                    cond_a = cond | cond_a
            ids = np.where(cond_a, ids, -1)
            scores = np.where(cond_a, scores, -1)
            bboxes = np.where(cond_a, bboxes, -1)
        ax = gcv.utils.viz.plot_bbox(img, bboxes, scores, ids, thresh=args.thresh,
                                     class_names=net.classes, ax=ax)
plt.show()
