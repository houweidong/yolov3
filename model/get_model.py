import os
import warnings
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from gluoncv.model_zoo.yolo.darknet import _conv2d, darknet53
from gluoncv.model_zoo.yolo.yolo_target import YOLOV3TargetMerger
# from gluoncv.loss import YOLOV3Loss
from loss import SelfLoss
from self_target import SelfDynamicTargetGeneratorSimple
from utils import get_order_config
from collections import OrderedDict

from gluoncv.model_zoo.yolo.yolo3 import _upsample, YOLODetectionBlockV3
from gluoncv.model_zoo.yolo import yolo3_darknet53_custom, yolo3_mobilenet1_0_coco, yolo3_mobilenet1_0_custom


class YOLOOutputV3(gluon.HybridBlock):
    """YOLO output layer V3.
    Parameters
    ----------
    index : int
        Index of the yolo output layer, to avoid naming conflicts only.
    num_class : int
        Number of foreground objects.
    anchors : iterable
        The anchor setting. Reference: https://arxiv.org/pdf/1804.02767.pdf.
    stride : int
        Stride of feature map.
    alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, Scalar, etc.
    """

    def __init__(self, index, num_class, anchors, stride, ignore_iou_thresh,
                 coop_config=(1,), alloc_size=(128, 128), label_smooth=True, **kwargs):
        super(YOLOOutputV3, self).__init__(**kwargs)
        anchors = np.array(anchors).astype('float32')
        self._xywho_num = len(coop_config)
        self._classes = num_class
        self._num_pred = self._xywho_num * (1 + 4) + num_class  # 1 objness + 4 box + num_class
        self._num_anchors = anchors.size // 2
        self._stride = stride
        dic = {32: slice(0, 1), 16: slice(1, 5), 8: slice(5, 21)}
        self._loss = SelfLoss(self._classes, ignore_iou_thresh, coop_config, dic[self._stride], label_smooth)
        # self._offset_t = (32 // stride) ** 2
        with self.name_scope():
            all_pred = self._num_pred * self._num_anchors
            self.prediction = nn.Conv2D(all_pred, kernel_size=1, padding=0, strides=1)
            # anchors will be multiplied to predictions
            anchors = anchors.reshape(1, 1, -1, 1, 2)
            self.anchors = self.params.get_constant('anchor_%d' % (index), anchors)
            # offsets will be added to predictions
            grid_x = np.arange(alloc_size[1])
            grid_y = np.arange(alloc_size[0])
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            # stack to (n, n, 2)
            offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)
            # expand dims to (1, 1, n, n, 2) so it's easier for broadcasting
            offsets = np.expand_dims(np.expand_dims(offsets, axis=0), axis=0)
            self.offsets = self.params.get_constant('offset_%d' % (index), offsets)
            horizontal_sig_levels = np.array(coop_config)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
            self.horizontal_sig_levels = self.params.get_constant('sig_level_%d' % (index), horizontal_sig_levels)

    def reset_class(self, classes, reuse_weights=None):
        """Reset class prediction.
        Parameters
        ----------
        classes : type
            Description of parameter `classes`.
        reuse_weights : dict
            A {new_integer : old_integer} mapping dict that allows the new predictor to reuse the
            previously trained weights specified by the integer index.
        Returns
        -------
        type
            Description of returned object.
        """
        self._clear_cached_op()
        # keep old records
        old_classes = self._classes
        old_pred = self.prediction
        old_num_pred = self._num_pred
        ctx = list(old_pred.params.values())[0].list_ctx()
        self._classes = len(classes)
        self._num_pred = 1 + 4 + len(classes)
        all_pred = self._num_pred * self._num_anchors
        # to avoid deferred init, number of in_channels must be defined
        in_channels = list(old_pred.params.values())[0].shape[1]
        self.prediction = nn.Conv2D(
            all_pred, kernel_size=1, padding=0, strides=1,
            in_channels=in_channels, prefix=old_pred.prefix)
        self.prediction.initialize(ctx=ctx)
        if reuse_weights:
            new_pred = self.prediction
            assert isinstance(reuse_weights, dict)
            for old_params, new_params in zip(old_pred.params.values(), new_pred.params.values()):
                old_data = old_params.data()
                new_data = new_params.data()
                for k, v in reuse_weights.items():
                    if k >= self._classes or v >= old_classes:
                        warnings.warn("reuse mapping {}/{} -> {}/{} out of range".format(
                            k, self._classes, v, old_classes))
                        continue
                    for i in range(self._num_anchors):
                        off_new = i * self._num_pred
                        off_old = i * old_num_pred
                        # copy along the first dimension
                        new_data[1 + 4 + k + off_new] = old_data[1 + 4 + v + off_old]
                        # copy non-class weights as well
                        new_data[off_new: 1 + 4 + off_new] = old_data[off_old: 1 + 4 + off_old]
                # set data to new conv layers
                new_params.set_data(new_data)

    def hybrid_forward(self, F, x, *args, anchors, offsets, horizontal_sig_levels):
        """Hybrid Forward of YOLOV3Output layer.
        Parameters
        ----------
        F : mxnet.nd or mxnet.sym
            `F` is mxnet.sym if hybridized or mxnet.nd if not.
        x : mxnet.nd.NDArray
            Input feature map.
        anchors : mxnet.nd.NDArray
            Anchors loaded from self, no need to supply.
        offsets : mxnet.nd.NDArray
            Offsets loaded from self, no need to supply.
        horizontal_sig_levels : mxnet.nd.NDArray
            horizontal_sig_levels loaded from self, no need to supply.
        Returns
        -------
        (tuple of) mxnet.nd.NDArray
            During training, return (bbox, raw_box_centers, raw_box_scales, objness,
            class_pred, anchors, offsets).
            During inference, return detections.
        """
        # prediction flat to (batch, pred per pixel, height * width)
        pred = self.prediction(x).reshape((0, self._num_anchors * self._num_pred, -1))
        # transpose to (batch, height * width, num_anchor, num_pred)
        pred = pred.transpose(axes=(0, 2, 1)).reshape((0, -1, self._num_anchors, self._num_pred))
        # components
        # transpose to (batch, height * width, num_anchor, num_xywho, 5)
        xywho = pred.slice_axis(axis=-1, begin=0, end=-self._classes).reshape((0, 0, 0, self._xywho_num, 5))
        # raw_box_centers = pred.slice_axis(axis=-1, begin=0, end=2)
        # raw_box_scales = pred.slice_axis(axis=-1, begin=2, end=4)
        # objness = pred.slice_axis(axis=-1, begin=4, end=5)
        # class_pred = pred.slice_axis(axis=-1, begin=5, end=None)
        class_pred = pred.slice_axis(axis=-1, begin=-self._classes, end=None)
        # ctrs = F.broadcast_mul(F.sigmoid(xywho.slice_axis(axis=-1, begin=0, end=2)), horizontal_sig_levels)
        ctrs = xywho.slice_axis(axis=-1, begin=0, end=2)
        raw_box_scales = xywho.slice_axis(axis=-1, begin=2, end=4)
        objness = xywho.slice_axis(axis=-1, begin=4, end=None)

        # valid offsets, (1, 1, height, width, 2)
        offsets = F.slice_like(offsets, x * 0, axes=(2, 3))
        # reshape to (1, height*width, 1, 2)
        offsets = F.broadcast_sub(offsets.reshape((1, -1, 1, 2)).expand_dims(-2) + 0.5, 0.5 * horizontal_sig_levels)

        # box_centers = F.broadcast_add(ctrs, offsets) * self._stride
        box_centers = F.broadcast_add(F.broadcast_mul(F.sigmoid(ctrs), horizontal_sig_levels), offsets) * self._stride
        box_scales = F.broadcast_mul(F.exp(raw_box_scales), anchors)
        confidence = F.sigmoid(objness)
        # class_score = F.broadcast_mul(F.sigmoid(class_pred), confidence)
        class_score = F.broadcast_mul(F.sigmoid(class_pred).expand_dims(-2), confidence)
        wh = box_scales / 2.0
        bbox = F.concat(box_centers - wh, box_centers + wh, dim=-1)

        # NDarray (h*w, anchor_num, xywho_num)
        # shape_array = F.shape_array(bbox).expand_dims(0)

        if autograd.is_training():
            # during training, we don't need to convert whole bunch of info to detection results
            if autograd.is_recording():
                # all_preds = [p.reshape((0, -1, 0)) for p in [objness, ctrs, raw_box_scales, class_pred, bbox]]
                objness = objness.reshape((0, -1, 1))
                ctrs = ctrs.reshape((0, -1, 2))
                raw_box_scales = ctrs.reshape((0, -1, 2))
                class_pred = class_pred.reshape((0, -3, -1))
                bbox = bbox.reshape((0, -1, 4))
                all_preds = [objness, ctrs, raw_box_scales, class_pred, bbox]
                loss_list = self._loss(*all_preds, *args)
                return loss_list, self._loss.order_sig_config
            else:
                return anchors, offsets

        # # prediction per class
        # bboxes = F.tile(bbox, reps=(self._classes, 1, 1, 1, 1))
        # scores = F.transpose(class_score, axes=(3, 0, 1, 2)).expand_dims(axis=-1)
        # ids = F.broadcast_add(scores * 0, F.arange(0, self._classes).reshape((0, 1, 1, 1, 1)))
        # detections = F.concat(ids, scores, bboxes, dim=-1)
        # # reshape to (B, xx, 6)
        # detections = F.reshape(detections.transpose(axes=(1, 0, 2, 3, 4)), (0, -1, 6))
        # prediction per class
        bboxes = F.tile(bbox, reps=(self._classes, 1, 1, 1, 1, 1))
        scores = F.transpose(class_score, axes=(4, 0, 1, 2, 3)).expand_dims(axis=-1)
        ids = F.broadcast_add(scores * 0, F.arange(0, self._classes).reshape((0, 1, 1, 1, 1, 1)))
        detections = F.concat(ids, scores, bboxes, dim=-1)
        # reshape to (B, xx, 6)
        detections = F.reshape(detections.transpose(axes=(1, 0, 2, 3, 4, 5)), (0, -1, 6))
        return detections


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
    """

    def __init__(self, stages, channels, anchors, strides, classes, alloc_size=(128, 128),
                 nms_thresh=0.45, nms_topk=400, post_nms=100, pos_iou_thresh=1.0,
                 ignore_iou_thresh=0.7, norm_layer=BatchNorm, norm_kwargs=None,
                 coop_configs=((1,), (1,), (1,)), label_smooth=True, **kwargs):
        super(YOLOV3, self).__init__(**kwargs)
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
        # if pos_iou_thresh >= 1:
        #     self._target_generator = YOLOV3TargetMerger(len(classes), ignore_iou_thresh)
        # else:
        #     raise NotImplementedError(
        #         "pos_iou_thresh({}) < 1.0 is not implemented!".format(pos_iou_thresh))
        # self._loss = SelfLoss(self._num_class, self._ignore_iou_thresh, self._coop_configs[::-1])
        with self.name_scope():
            self.stages = nn.HybridSequential()
            self.transitions = nn.HybridSequential()
            self.yolo_blocks = nn.HybridSequential()
            self.yolo_outputs = nn.HybridSequential()
            # note that anchors and strides should be used in reverse order
            for i, stage, channel, anchor, stride, coop_config in zip(
                    range(len(stages)), stages, channels, anchors[::-1], strides[::-1], coop_configs[::-1]):
                self.stages.add(stage)
                block = YOLODetectionBlockV3(
                    channel, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                self.yolo_blocks.add(block)
                output = YOLOOutputV3(i, len(classes), anchor, stride, self._ignore_iou_thresh,
                                      coop_config=coop_config, alloc_size=alloc_size, label_smooth=label_smooth)
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

        loss = OrderedDict()
        for sig_level in self._order_sig_config:
            loss['obj_sig{}'.format(sig_level)] = []
            loss['xy_sig{}'.format(sig_level)] = []
            loss['wh_sig{}'.format(sig_level)] = []
        loss['cls'] = []

        # the YOLO output layers are used in reverse order, i.e., from very deep layers to shallow
        for i, block, output in zip(range(len(routes)), self.yolo_blocks, self.yolo_outputs):
            x, tip = block(x)
            if autograd.is_training():

                if autograd.is_recording():
                    loss_list, coop_config = output(tip, *args)
                    for sig_level in coop_config:
                        loss['obj_sig{}'.format(sig_level)].append(loss_list.pop(0))
                        loss['xy_sig{}'.format(sig_level)].append(loss_list.pop(0))
                        loss['wh_sig{}'.format(sig_level)].append(loss_list.pop(0))
                    loss['cls'].append(loss_list.pop(0))
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
                # generate losses and return them directly
                for key, item in loss.items():
                    loss[key] = sum(item)

                return [loss[l] for l in loss]

            # this is only used in DataLoader transform function.
            return all_anchors, all_offsets, all_feat_maps

        # concat all detection results from different stages
        result = F.concat(*all_detections, dim=1)
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

    # def reset_stride(self, hXw):
    #     self._loss_stride = [(hXw * 3) // (std ** 2) for std in self._strides]
    #     self._loss.set_stride(self._loss_stride)


def get_yolov3(name, stages, filters, anchors, strides, classes, coop_configs,
               dataset, pretrained=False, ctx=mx.cpu(),
               root=os.path.join('~', '.mxnet', 'models'), label_smooth=True, **kwargs):
    """Get YOLOV3 models.
    Parameters
    ----------
    name : str or None
        Model name, if `None` is used, you must specify `features` to be a `HybridBlock`.
    stages : iterable of str or `HybridBlock`
        List of network internal output names, in order to specify which layers are
        used for predicting bbox values.
        If `name` is `None`, `features` must be a `HybridBlock` which generate multiple
        outputs for prediction.
    filters : iterable of float or None
        List of convolution layer channels which is going to be appended to the base
        network feature extractor. If `name` is `None`, this is ignored.
    sizes : iterable fo float
        Sizes of anchor boxes, this should be a list of floats, in incremental order.
        The length of `sizes` must be len(layers) + 1. For example, a two stage SSD
        model can have ``sizes = [30, 60, 90]``, and it converts to `[30, 60]` and
        `[60, 90]` for the two stages, respectively. For more details, please refer
        to original paper.
    ratios : iterable of list
        Aspect ratios of anchors in each output layer. Its length must be equals
        to the number of SSD output layers.
    steps : list of int
        Step size of anchor boxes in each output layer.
    classes : iterable of str
        Names of categories.
    dataset : str
        Name of dataset. This is used to identify model name because models trained on
        different datasets are going to be very different.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    HybridBlock
        A YOLOV3 detection network.
    """
    net = YOLOV3(stages, filters, anchors, strides,
                 classes=classes, coop_configs=coop_configs, label_smooth=label_smooth, **kwargs)
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        full_name = '_'.join(('yolo3', name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx)
    return net


def yolo3_darknet53_coco(pretrained_base=True, pretrained=False,
                         norm_layer=BatchNorm, norm_kwargs=None,
                         coop_configs=((1,), (1,), (1,)), label_smooth=True, **kwargs):
    """YOLO3 multi-scale with darknet53 base network on COCO dataset.
    Parameters
    ----------
    pretrained_base : boolean
        Whether fetch and load pretrained weights for base network.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    coop_configs : tuple
        Sigmoid configs
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
    strides = [8, 16, 32]
    classes = COCODetection.CLASSES
    return get_yolov3(
        'darknet53', stages, [512, 256, 128], anchors, strides, classes, coop_configs, 'coco',
        pretrained=pretrained, norm_layer=norm_layer, norm_kwargs=norm_kwargs, label_smooth=label_smooth, **kwargs)


_models = {
    'yolo3_darknet53_coco': yolo3_darknet53_coco,
    'yolo3_darknet53_custom': yolo3_darknet53_custom,
    'yolo3_mobilenet1.0_coco': yolo3_mobilenet1_0_coco,
    'yolo3_mobilenet1.0_custom': yolo3_mobilenet1_0_custom,
}


def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

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
