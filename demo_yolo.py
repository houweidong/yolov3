
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
from model.utils import config
import cv2
from mxnet import nd
from utils.timer import Timer


def w2f(image_d, ctx, args, img_bbox_d):
    if os.path.isdir(image_d):

        for image_dd in os.listdir(image_d):
            dir = image_d.split('/')[-1]
            try:
                w2f(os.path.join(image_d, image_dd), ctx, args, os.path.join(img_bbox_d, dir))
            except:
                continue

    else:

        # img_bbox_f = open(os.path.join(img_bbox_d, image_d.split('/')[-1] + '.txt'), 'a')
        img_bbox_f = open(img_bbox_d + '.txt', 'a')
        image = image_d
        try:
            ids, scores, bboxes, img = forward(image, ctx, args)
        except:
            raise Exception('valid picture')
        else:
            image_name = image.split('/')[-1]
            bboxes_str = ''
            for i, coord in enumerate(np.reshape(bboxes, -1)):
                # if i != 0 and i % 4 == 0:
                #     bboxes_str += ', '
                bboxes_str += str(coord) + ' '
            write2file = image_name + ' ' + bboxes_str + '\n'
            img_bbox_f.write(write2file)
            img_bbox_f.close()


def forward(image_p, ctx, args):

    try:
        if not args.demo:
            image = mx.image.imread(image_p)
    except:
        print(image_p)
        os.remove(image_p)
        raise Exception('valid picture')
    else:
        max_len = max(image.shape[0], image.shape[1])
        min_len = min(image.shape[0], image.shape[1])
        # short = int(args.short)
        short = 320
        x, _ = presets.yolo.transform_test(image, short=short, max_size=1024)
        scale = min_len / short
        if max_len / scale > 1024:
            scale = max_len / 1024
        img = image.asnumpy().astype('uint8')

        x = x.as_in_context(ctx[0])
        ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
        bboxes = bboxes * scale
        # select args.classes in ids
        if args.classes:
            cla_ids = []
            # str.strip
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

            cond_a = (scores >= args.thresh) & cond_a
        else:
            cond_a = scores >= args.thresh
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
    parser.add_argument('--images', type=str, default='nanzhuang',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.2,
                        help='Threshold of object score when visualize the bboxes.')
    parser.add_argument('--classes', type=str, default='person',
                        help='classes to be displayed. could more than one class')
    parser.add_argument('--coop-cfg', type=str, default='2, 2, 2',
                        help='coop configs. "," separate different output head, '
                             '" " separate different sig level in a same output layer. '
                             'such as 1,2 3 4,1 2 3')
    parser.add_argument('--nms-mode', type=str, default='Default', choices=['Default', 'Exclude', 'Merge'])
    parser.add_argument('--demo', action='store_true', help='whether to use camera or video.')
    parser.add_argument('--video', default=None, help='path to vedio')
    parser.add_argument('--w2f', action='store_true', help='whether to dump to the file')
    parser.add_argument('--root', type=str, default='/root/dataset/', help='dataset root')
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
    elif '.' not in args.images:
        # for file in os.listdir(args.root):
        #     print(file)
        # print(os.listdir(args.root))
        val_data = os.path.join(args.root, args.images)
        image_list = [os.path.join(val_data, img) for img in os.listdir(val_data)]
    else:
        image_list = [os.path.join('./data', x.strip()) for x in args.images.split(',') if x.strip()]

    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(args.network, pretrained=True)
    else:
        config(args)
        net = get_model(args.network, pretrained_base=False, coop_configs=args.coop_cfg, nms_mode=args.nms_mode)
        net.initialize()
        net.load_parameters(args.pretrained, allow_missing=True, ignore_extra=True)
        # print('wo')
    net.set_nms(0.5, 200)
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
        if args.w2f:
            # we write the file to the fixed dir /root/dataset/results
            # img_bbox_f = open(os.path.join('/root/dataset/results', args.images+'.txt'),'w')
            img_bbox_d = os.path.join('/root/dataset/results', args.images)

            isExists = os.path.exists(img_bbox_d)

            if not isExists:
                os.makedirs(img_bbox_d)

        for image in image_list:

            # ids, scores, bboxes, img = forward(image, ctx, args)
            if args.w2f:
                w2f(image, ctx, args, img_bbox_d)
                # image_name = image.split('/')[-1]
                # bboxes_str = ''
                # for i, coord in enumerate(np.reshape(bboxes, -1)):
                #     # if i != 0 and i % 4 == 0:
                #     #     bboxes_str += ', '
                #     bboxes_str += str(coord) + ' '
                # write2file = image_name + ' ' + bboxes_str + '\n'
                # img_bbox_f.write(write2file)
            else:
                ids, scores, bboxes, img = forward(image, ctx, args)
                ax = None
                ax = gcv.utils.viz.plot_bbox(img, bboxes, scores, ids, thresh=args.thresh,
                                             class_names=net.classes, ax=ax)
                plt.show()
        # if args.w2f:
        #     img_bbox_f.close()

