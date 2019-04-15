import mxnet as mx
# from math import pi, cos
# from mxnet import nd
import numpy as np
from gluoncv.data.batchify import Pad

#
# def get_order_config(coop_configs):
#     order_config = []
#     for configs in coop_configs:
#         for config in configs:
#             if config not in order_config:
#                 order_config.append(config)
#
#     return sorted(order_config)


def config(args):

    def cfg(cfg_str):
        cfg_list = []
        configs = list(filter(None, cfg_str.split(',')))
        if len(configs) == 1:
            configs = configs * 3
        if len(configs) != 3:
            raise Exception('coop configs should have three layers!')
        for config in configs:
            # coop_configs.append(tuple(sorted(map(int, filter(None, config.split(' '))))))
            config = list(map(float, filter(None, config.split(' '))))
            if len(config) < 3:
                config = config * 3
            else:
                config = np.array(config).reshape(-1, 3).transpose()
            cfg_list.append(config)
        return np.array(cfg_list).reshape((3, 3, -1))

    args.coop_cfg = cfg(args.coop_cfg)
    args.margin = cfg(args.margin) if hasattr(args, 'margin') else None
    args.thre_cls = cfg(args.thre_cls) if hasattr(args, 'thre_cls') else None
    args.sigma = cfg(args.sigma) if hasattr(args, 'sigma-weight') else None


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, a_min=0, a_max=None) * \
                 np.clip(inter_rect_y2 - inter_rect_y1, a_min=0, a_max=None)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def self_box_nms(data, overlap_thresh=0.45, valid_thresh=0.01, topk=None, nms_style='Default'):
    """
    Removes detections with lower object confidence score than 'overlap_thresh'
    Different class bboxes without interference

    Parameters
    ----------
    data : NDArray
        The input with shape (ids, scores, boxes), and the boxes is in corner mode
    overlap_thresh : float, optional, default=0.5
        Overlapping(IoU) threshold to suppress object with smaller score.
    valid_thresh : float, optional, default=0
        Filter input boxes to those whose scores greater than valid_thresh.
    topk : int, optional, default='-1'
        Apply nms to topk boxes with descending scores, -1 to no restriction.
    nms_style : nms mode
        'Default', 'Exclude', 'Merge'

    Returns
    -------
    out : NDArray or list of NDArrays
        The output with shape (ids, scores, boxes), and the boxes is in corner mode
    """

    data = data.asnumpy()
    output = [-np.ones(shape=(1, 6)) for _ in range(len(data))]

    for image_i, pred in enumerate(data):
        # Sort the predicted boxes by maximum  confidence
        v = (pred[:, 1] > valid_thresh).nonzero()[0]
        pred = pred[v]
        conf_sort_index = np.argsort(pred[:, 1])[::-1]
        pred = pred[conf_sort_index]
        if topk:
            pred = pred[:topk]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue

        unique_labels = np.unique(pred[:, 0])
        for c in unique_labels:
            # Get the predicted boxes with class c
            pc = pred[pred[:, 0] == c]
            # Non-maximum suppression
            det_max = []
            if nms_style == 'Default':
                while pc.shape[0]:
                    det_max.append(pc[:1])  # save highest conf detection
                    if len(pc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(pc[:1, 2:], pc[1:, 2:])  # iou with other boxes
                    pc = pc[1:][iou < overlap_thresh]  # remove ious > threshold

            elif nms_style == 'Exclude':  # requires overlap, single boxes erased
                while len(pc) > 1:
                    iou = bbox_iou(pc[:1, 2:], pc[1:, 2:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(pc[:1])
                    pc = pc[1:][iou < overlap_thresh]  # remove ious > threshold

            elif nms_style == 'Merge':  # weighted mixture box
                while len(pc) > 0:
                    iou = bbox_iou(pc[:1, 2:], pc[0:, 2:])  # iou with other boxes
                    i = iou > overlap_thresh

                    weights = pc[i, 1:2]
                    pc[0, 2:] = (weights * pc[i, 2:]).sum(0) / weights.sum()
                    det_max.append(pc[:1])
                    pc = pc[iou < overlap_thresh]

            if len(det_max) > 0:
                # Add max detections to outputs
                output[image_i] = np.concatenate((output[image_i], np.concatenate(det_max)))
    result = Pad(axis=0, pad_val=-1)(output)
    ids = result.slice_axis(axis=-1, begin=0, end=1)
    scores = result.slice_axis(axis=-1, begin=1, end=2)
    bboxes = result.slice_axis(axis=-1, begin=2, end=None)
    return ids, scores, bboxes


class LossMetric:

    def __init__(self, order_sig_config):
        self._order_sig_config = order_sig_config
        for sig_level in self._order_sig_config:
            self.__dict__['obj_sig{}'.format(sig_level)] = mx.metric.Loss('obj_sig{}'.format(sig_level))
            self.__dict__['xy_sig{}'.format(sig_level)] = mx.metric.Loss('xy_sig{}'.format(sig_level))
            self.__dict__['wh_sig{}'.format(sig_level)] = mx.metric.Loss('wh_sig{}'.format(sig_level))

        self.__dict__['cls'] = mx.metric.Loss('cls')

    def initial(self):
        for sig_level in self._order_sig_config:
            self.__dict__['obj_sig{}list'.format(sig_level)] = []
            self.__dict__['xy_sig{}list'.format(sig_level)] = []
            self.__dict__['wh_sig{}list'.format(sig_level)] = []
        self.__dict__['cls_list'] = []

    def append(self, loss_list):
        index = 0
        for sig_level in self._order_sig_config:
            self.__dict__['obj_sig{}list'.format(sig_level)].append(loss_list[index])
            self.__dict__['xy_sig{}list'.format(sig_level)].append(loss_list[index + 1])
            self.__dict__['wh_sig{}list'.format(sig_level)].append(loss_list[index + 2])
            index += 3
        self.__dict__['cls_list'].append(loss_list[-1])

    def update(self):
        for sig_level in self._order_sig_config:
            self.__dict__['obj_sig{}'.format(sig_level)].update(0, self.__dict__['obj_sig{}list'.format(sig_level)])
            self.__dict__['xy_sig{}'.format(sig_level)].update(0, self.__dict__['xy_sig{}list'.format(sig_level)])
            self.__dict__['wh_sig{}'.format(sig_level)].update(0, self.__dict__['wh_sig{}list'.format(sig_level)])
        self.__dict__['cls'].update(0, self.__dict__['cls_list'])

    def get(self):
        name_loss = []
        name_loss_str = ''
        for sig_level in self._order_sig_config:
            name_loss_str += ', '
            name1, loss1 = self.__dict__['obj_sig{}'.format(sig_level)].get()
            name2, loss2 = self.__dict__['xy_sig{}'.format(sig_level)].get()
            name3, loss3 = self.__dict__['wh_sig{}'.format(sig_level)].get()
            name_loss += [name1, loss1, name2, loss2, name3, loss3]
            name_loss_str += '{}={:.3f}, {}={:.3f}, {}={:.3f}'
        name4, loss4 = self.__dict__['cls'].get()
        name_loss += [name4, loss4]
        name_loss_str += ', {}={:.3f}'
        return name_loss_str, name_loss


# simple indicate not to distinguish level
class LossMetricSimple:

    def __init__(self, coop_loss):
        name = ['ObjLoss', 'BoxCenterLoss', 'BoxScaleLoss', 'ClassLoss', 'CoopBoxCenterLoss', 'CoopBoxScaleLoss'] \
            if coop_loss else ['ObjLoss', 'BoxCenterLoss', 'BoxScaleLoss', 'ClassLoss']
        self.metrics_list = []
        self.len_metrics = len(name)
        for i in range(self.len_metrics):
            self.metrics_list.append(mx.metric.Loss(name[i]))

        self.loss_list = [[]] * self.len_metrics

    def initial(self):
        for i in range(self.len_metrics):
            self.loss_list[i] = []

    def append(self, loss_list):
        for i in range(self.len_metrics):
            self.loss_list[i].append(loss_list[i])

    def update(self):
        for i in range(self.len_metrics):
            self.metrics_list[i].update(0, self.loss_list[i])

    def get(self):
        name_loss = []
        name_loss_str = ''
        for i in range(self.len_metrics):
            name_i, loss_i = self.metrics_list[i].get()
            name_loss += [name_i, loss_i]
            name_loss_str += '{}={:.3f} '
        return name_loss_str, name_loss

