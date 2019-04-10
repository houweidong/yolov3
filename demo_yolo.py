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
import cv2
from mxnet import nd
from utils.timer import Timer


def forward(image, ctx, args):
    if not args.demo:
        x, img = presets.yolo.load_test(image, short=512)
    else:
        x, img = presets.yolo.transform_test(image, short=512)
    x = x.as_in_context(ctx[0])
    ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
    # select args.classes in ids
    if args.classes:
        cla_ids = []
        str.strip
        clses = list(map(str.strip, filter(None, args.classes.split(','))))
        cond_a = np.zeros_like(ids, dtype=np.bool)
        for cls in clses:
            cla_ids.append(COCODetection.CLASSES.index(cls))
            for cid in cla_ids:
                cond = cid == ids.astype(np.int)
                cond_a = cond | cond_a
        # ids = np.where(cond_a, ids, -1)
        # scores = np.where(cond_a, scores, -1)
        # bboxes = np.where(cond_a, bboxes, -1)
        ids = ids[cond_a.reshape(-1)]
        scores = scores[cond_a.reshape(-1)]
        bboxes = bboxes[cond_a.reshape(-1)]
    # print(ids)
    # print(scores)
    # print(bboxes)
    return ids, scores, bboxes, img


def draw_result(ids, scores, bboxes, img):

    for i in range(len(ids)):

        if scores[i, 0] < args.thresh:
            continue
        x1, y1, x2, y2 = int(bboxes[i, 0]), int(bboxes[i, 1]), int(bboxes[i, 2]), int(bboxes[i, 3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x2, y1), (125, 125, 125), -1)
        lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
        cv2.putText(
            img, str(ids[i, 0]) + ' : %.2f' % scores[i, 0],
            (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 1, lineType)


def parse_args():
    parser = argparse.ArgumentParser(description='Test with YOLO networks.')
    parser.add_argument('--network', type=str, default='yolo3_darknet53_coco',
                        help="Base network name")
    parser.add_argument('--images', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.05,
                        help='Threshold of object score when visualize the bboxes.')
    parser.add_argument('--classes', type=str, default='',
                        help='classes to be displayed. could more than one class')
    parser.add_argument('--coop-cfg', type=str, default='2, 2, 2',
                        help='coop configs. "," separate different output head, '
                             '" " separate different sig level in a same output layer. '
                             'such as 1,2 3 4,1 2 3')
    parser.add_argument('--nms-mode', type=str, default='Default', choices=['Default', 'Exclude', 'Merge'])
    parser.add_argument('--demo', action='store_true', help='whether to use camera or video.')
    parser.add_argument('--video', default=None, help='path to vedio')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    # grab some image if not specified
    if not args.images.strip():
        val_data = '/root/dataset/coco2017/train2017'
        image_list = [os.path.join(val_data, img) for img in os.listdir(val_data)]

    else:
        image_list = [os.path.join('./data', x.strip()) for x in args.images.split(',') if x.strip()]

    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(args.network, pretrained=True)
    else:
        coop_cfg = get_coop_config(args.coop_cfg)
        net = get_model(args.network, pretrained_base=False, coop_configs=coop_cfg, nms_mode=args.nms_mode)
        net.initialize()
        net.load_parameters(args.pretrained, allow_missing=True, ignore_extra=True)
        # print('wo')
    net.set_nms(0.99, 200)
    net.collect_params().reset_ctx(ctx=ctx)

    if args.demo:
        if args.video is not None:
            print('Reading vedio from {}'.format(args.video))
            cap = cv2.VideoCapture(args.video)

        else:
            print('Realtime detectiong')
            cap = cv2.VideoCapture(0)

        detect_timer = Timer()
        while True:

            ret, frame = cap.read()
            detect_timer.tic()

            # to transform frame to ndarray
            # frame = nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32))
            frame = nd.array(frame)
            ids, scores, bboxes, img = forward(frame, ctx, args)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(
                detect_timer.average_time))

            draw_result(ids, scores, bboxes, img)
            cv2.imshow('Camera', img)
            cv2.waitKey(10)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        for image in image_list:
            ax = None
            ids, scores, bboxes, img = forward(image, ctx, args)
            ax = gcv.utils.viz.plot_bbox(img, bboxes, scores, ids, thresh=args.thresh,
                                         class_names=net.classes, ax=ax)
            plt.show()
