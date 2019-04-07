import os
import warnings
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
# from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from gluoncv.model_zoo.yolo.darknet import _conv2d, darknet53
from gluoncv.model_zoo.yolo.yolo_target import YOLOV3TargetMerger
# from model.loss import SelfLoss
from model.utils import get_order_config
from model.model_utils import YOLOOutputV3
from gluoncv.model_zoo.yolo.yolo3 import _upsample, YOLODetectionBlockV3
from gluoncv.model_zoo.yolo import yolo3_darknet53_custom, yolo3_mobilenet1_0_coco, yolo3_mobilenet1_0_custom


class YOLOV3(gluon.HybridBlock):
    """YOLO V3 detection network.
    Reference: https://arxiv.org/pdf/1804.02767.pdf.
    Parameters
    ----------
    stages : mxnet.gluon.HybridBlock
        Staged feature extraction blocks.
        For example, 3 stages and 3 YOLO output layers are used original paper.
    channels : iterable
        Number of conv channels for each appended stage.
        `len(channels)` should match `len(stages)`.
    num_class : int
        Number of foreground objects.
    anchors : iterable
        The anchor setting. `len(anchors)` should match `len(stages)`.
    strides : iterable
        Strides of feature map. `len(strides)` should match `len(stages)`.
    alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, Scalar, etc.
    nms_thresh : float, default is 0.45.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    pos_iou_thresh : float, default is 1.0
        IOU threshold for true anchors that match real objects.
        'pos_iou_thresh < 1' is not implemented.
    ignore_iou_thresh : float
        Anchors that has IOU in `range(ignore_iou_thresh, pos_iou_thresh)` don't get
        penalized of objectness score.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    nms_mode : string
        "Default" : default nms
        "Exclude" : Exclude the boxes has no iou with other boxes larger than the ignore_iou_thresh
        "Merge" : Merge all boxes between which iou larger than ignore_iou_thresh according to the probability
    coop_mode : string
        "flat", different level grids have same weight loss in the training phase
        "convex", the center grids have higher weight than the marginal grids in the training phase
        "concave", the marginal grids have higher weight than the center grids in the training phase
    sigma_weight : float
        for coop_mode params, we use Gaussian distribution to generate the weights according to the grid level,
        the sigma_weight is the distribution's params
    """

    def __init__(self, stages, channels, anchors, strides, classes, alloc_size=(128, 128), nms_thresh=0.45, nms_topk=400,
                 post_nms=100, pos_iou_thresh=1.0, ignore_iou_thresh=0.7,norm_layer=BatchNorm, norm_kwargs=None,
                 coop_configs=((1,), (1,), (1,)), label_smooth=True, nms_mode='Default', coop_mode='flat',
                 sigma_weight=1.6, specific_anchor='default', sa_level=1, kernels=None, coop_loss=False, **kwargs):
        super(YOLOV3, self).__init__(**kwargs)
        assert nms_mode in ['Default', 'Merge', 'Exclude']
        self._coop_configs = coop_configs
        self._order_sig_config = get_order_config(coop_configs)
        self._classes = classes
        self._num_class = len(classes)
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self._pos_iou_thresh = pos_iou_thresh
        self._ignore_iou_thresh = ignore_iou_thresh
        self._strides = strides[::-1]
        self._loss_stride = None
        self._nms_mode = nms_mode
        self._specific_anchor = specific_anchor
        self._sa_level = sa_level
        self._coop_loss = coop_loss

        with self.name_scope():
            self.stages = nn.HybridSequential()
            self.transitions = nn.HybridSequential()
            self.yolo_blocks = nn.HybridSequential()
            self.yolo_outputs = nn.HybridSequential()
            # note that anchors and strides should be used in reverse order
            for i, stage, channel, anchor, stride, coop_config, kernel in zip(
                    range(len(stages)), stages, channels, anchors[::-1], strides[::-1], coop_configs[::-1], kernels[::-1]):
                self.stages.add(stage)
                block = YOLODetectionBlockV3(channel, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                self.yolo_blocks.add(block)
                output = YOLOOutputV3(i, len(classes), anchor, stride, self._ignore_iou_thresh, coop_mode, sigma_weight,
                                      coop_config=coop_config, alloc_size=alloc_size, label_smooth=label_smooth,
                                      specific_anchor=specific_anchor, sa_level=sa_level, kernels=kernel, coop_loss=coop_loss,
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                self.yolo_outputs.add(output)
                if i > 0:
                    self.transitions.add(_conv2d(channel, 1, 0, 1,
                                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs))

    @property
    def num_class(self):
        """Number of (non-background) categories.
        Returns
        -------
        int
            Number of (non-background) categories.
        """
        return self._num_class

    @property
    def classes(self):
        """Return names of (non-background) categories.
        Returns
        -------
        iterable of str
            Names of (non-background) categories.
        """
        return self._classes

    def hybrid_forward(self, F, x, *args):
        """YOLOV3 network hybrid forward.
        Parameters
        ----------
        F : mxnet.nd or mxnet.sym
            `F` is mxnet.sym if hybridized or mxnet.nd if not.
        x : mxnet.nd.NDArray
            Input data.
        *args : optional, mxnet.nd.NDArray
            During training, extra inputs are required:
            (gt_boxes, obj_t, centers_t, scales_t, weights_t, clas_t)
            These are generated by YOLOV3PrefetchTargetGenerator in dataloader transform function.
        Returns
        -------
        (tuple of) mxnet.nd.NDArray
            During inference, return detections in shape (B, N, 6)
            with format (cid, score, xmin, ymin, xmax, ymax)
            During training, return losses only: (obj_loss, center_loss, scale_loss, cls_loss).
        """
        all_anchors = []
        all_offsets = []
        all_feat_maps = []
        all_detections = []
        # all_shape_arrays = []
        routes = []
        for stage, block, output in zip(self.stages, self.yolo_blocks, self.yolo_outputs):
            x = stage(x)
            routes.append(x)

        # loss = OrderedDict()
        loss = [0., 0., 0., 0.]

        # the YOLO output layers are used in reverse order, i.e., from very deep layers to shallow
        for i, block, output in zip(range(len(routes)), self.yolo_blocks, self.yolo_outputs):
            x, tip = block(x)
            if autograd.is_training():

                if autograd.is_recording():
                    obj_loss, center_loss, scale_loss, cls_loss = output(tip, *args)
                    loss[0] = loss[0] + obj_loss
                    loss[1] = loss[1] + center_loss
                    loss[2] = loss[2] + scale_loss
                    loss[3] = loss[3] + cls_loss
                else:
                    anchors, offsets = output(tip)
                    all_anchors.append(anchors)
                    all_offsets.append(offsets)
                    # here we use fake featmap to reduce memory consuption, only shape[2, 3] is used
                    fake_featmap = F.zeros_like(tip.slice_axis(
                        axis=0, begin=0, end=1).slice_axis(axis=1, begin=0, end=1))
                    all_feat_maps.append(fake_featmap)
            else:
                dets = output(tip)
                all_detections.append(dets)
            if i >= len(routes) - 1:
                break
            # add transition layers
            x = self.transitions[i](x)
            # upsample feature map reverse to shallow layers
            upsample = _upsample(x, stride=2)
            route_now = routes[::-1][i + 1]
            x = F.concat(F.slice_like(upsample, route_now * 0, axes=(2, 3)), route_now, dim=1)

        if autograd.is_training():
            # during training, the network behaves differently since we don't need detection results
            if autograd.is_recording():
                return loss

            # this is only used in DataLoader transform function.
            return all_anchors, all_offsets, all_feat_maps

        # concat all detection results from different stages
        result = F.concat(*all_detections, dim=1)
        if self._nms_mode == 'Default':
            # apply nms per class
            if 0 < self.nms_thresh < 1:
                result = F.contrib.box_nms(
                    result, overlap_thresh=self.nms_thresh, valid_thresh=0.01,
                    topk=self.nms_topk, id_index=0, score_index=1, coord_start=2, force_suppress=False)
                if self.post_nms > 0:
                    result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
            ids = result.slice_axis(axis=-1, begin=0, end=1)
            scores = result.slice_axis(axis=-1, begin=1, end=2)
            bboxes = result.slice_axis(axis=-1, begin=2, end=None)
            return ids, scores, bboxes
        else:
            # apply self nms out of the net, in the val func
            return result

    def set_nms(self, nms_thresh=0.45, nms_topk=400, post_nms=100):
        """Set non-maximum suppression parameters.
        Parameters
        ----------
        nms_thresh : float, default is 0.45.
            Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that every Detection
             result is used in NMS.
        post_nms : int, default is 100
            Only return top `post_nms` detection results, the rest is discarded. The number is
            based on COCO dataset which has maximum 100 objects per image. You can adjust this
            number if expecting more objects. You can use -1 to return all detections.
        Returns
        -------
        None
        """
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    def reset_class(self, classes, reuse_weights=None):
        """Reset class categories and class predictors.
        Parameters
        ----------
        classes : iterable of str
            The new categories. ['apple', 'orange'] for example.
        reuse_weights : dict
            A {new_integer : old_integer} or mapping dict or {new_name : old_name} mapping dict,
            or a list of [name0, name1,...] if class names don't change.
            This allows the new predictor to reuse the
            previously trained weights specified.

        Example
        -------
        >>> net = gluoncv.model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
        >>> # use direct name to name mapping to reuse weights
        >>> net.reset_class(classes=['person'], reuse_weights={'person':'person'})
        >>> # or use interger mapping, person is the 14th category in VOC
        >>> net.reset_class(classes=['person'], reuse_weights={0:14})
        >>> # you can even mix them
        >>> net.reset_class(classes=['person'], reuse_weights={'person':14})
        >>> # or use a list of string if class name don't change
        >>> net.reset_class(classes=['person'], reuse_weights=['person'])

        """
        self._clear_cached_op()
        old_classes = self._classes
        self._classes = classes
        if self._pos_iou_thresh >= 1:
            self._target_generator = YOLOV3TargetMerger(len(classes), self._ignore_iou_thresh)
        if isinstance(reuse_weights, (dict, list)):
            if isinstance(reuse_weights, dict):
                # trying to replace str with indices
                for k, v in reuse_weights.items():
                    if isinstance(v, str):
                        try:
                            v = old_classes.index(v)  # raise ValueError if not found
                        except ValueError:
                            raise ValueError(
                                "{} not found in old class names {}".format(v, old_classes))
                        reuse_weights[k] = v
                    if isinstance(k, str):
                        try:
                            new_idx = self._classes.index(k)  # raise ValueError if not found
                        except ValueError:
                            raise ValueError(
                                "{} not found in new class names {}".format(k, self._classes))
                        reuse_weights.pop(k)
                        reuse_weights[new_idx] = v
            else:
                new_map = {}
                for x in reuse_weights:
                    try:
                        new_idx = self._classes.index(x)
                        old_idx = old_classes.index(x)
                        new_map[new_idx] = old_idx
                    except ValueError:
                        warnings.warn("{} not found in old: {} or new class names: {}".format(
                            x, old_classes, self._classes))
                reuse_weights = new_map

        for outputs in self.yolo_outputs:
            outputs.reset_class(classes, reuse_weights=reuse_weights)


# to decide whether to pretrain yolov3
def get_yolov3(name, stages, filters, anchors, strides, classes, coop_configs, dataset, pretrained=False, ctx=mx.cpu(),
               root=os.path.join('~', '.mxnet', 'models'), nms_mode='Default', label_smooth=True, coop_mode='flat',
               sigma_weight=1.6, ignore_iou_thresh=0.7, specific_anchor='default', sa_level=1, kernels=None,
               coop_loss=False, **kwargs):
    net = YOLOV3(stages, filters, anchors, strides,
                 classes=classes, coop_configs=coop_configs, label_smooth=label_smooth, nms_mode=nms_mode,
                 coop_mode=coop_mode, sigma_weight=sigma_weight, ignore_iou_thresh=ignore_iou_thresh,
                 specific_anchor=specific_anchor, sa_level=sa_level, kernels=kernels, coop_loss=coop_loss, **kwargs)
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        full_name = '_'.join(('yolo3', name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx,
                            allow_missing=True, ignore_extra=True)
    return net


# to build the backbone for yolov3
def yolo3_darknet53_coco(pretrained_base=True, pretrained=False, norm_layer=BatchNorm, norm_kwargs=None,
                         coop_configs=((1,), (1,), (1,)), label_smooth=True, nms_mode='Default', coop_mode='flat',
                         sigma_weight=1.6, ignore_iou_thresh=0.7, specific_anchor='default', sa_level=1, sq_level=5,
                         coop_loss=False, **kwargs):
    """
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    from gluoncv.data import COCODetection
    pretrained_base = False if pretrained else pretrained_base
    base_net = darknet53(
        pretrained=pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    stages = [base_net.features[:15], base_net.features[15:24], base_net.features[24:]]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    rectangle = [[(3, 2), (6, 3), (4, 6)], [(4, 2), (3, 4), (8, 4)], [(3, 4), (7, 5), (10, 12)]]
    rectanglefix = [[(3, 2), (4, 2), (2, 3)], [(4, 2), (3, 4), (4, 2)], [(3, 4), (5, 4), (5, 6)]]
    square = [sq_level] * 3  # just placeholder
    if specific_anchor == 'rectangle':
        kernels = rectangle
    elif specific_anchor =='rectanglefix':
        kernels = rectanglefix
    else:
        kernels = square
    strides = [8, 16, 32]
    classes = COCODetection.CLASSES
    return get_yolov3(
        'darknet53', stages, [512, 256, 128], anchors, strides, classes, coop_configs, 'coco', pretrained=pretrained,
        norm_layer=norm_layer, norm_kwargs=norm_kwargs, label_smooth=label_smooth, nms_mode=nms_mode, coop_mode=coop_mode,
        sigma_weight=sigma_weight, ignore_iou_thresh=ignore_iou_thresh, specific_anchor=specific_anchor, sa_level=sa_level,
        kernels=kernels, coop_loss=coop_loss, **kwargs)


_models = {
    'yolo3_darknet53_coco': yolo3_darknet53_coco,
    'yolo3_darknet53_custom': yolo3_darknet53_custom,
    'yolo3_mobilenet1.0_coco': yolo3_mobilenet1_0_coco,
    'yolo3_mobilenet1.0_custom': yolo3_mobilenet1_0_custom,
}


def get_model(name, **kwargs):
    """Returns a pre-defined model by name
    Returns
    -------
    HybridBlock
        The model.
    """
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net
