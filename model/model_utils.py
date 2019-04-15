import warnings
import numpy as np
from mxnet import gluon
from mxnet import autograd
# from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from gluoncv.model_zoo.yolo.darknet import _conv2d
from model.loss import SelfLoss, SelfCoopLoss


def select_yolo_output(out_channel, specific_anchor, sa_level, kernel, norm_layer=BatchNorm, norm_kwargs=None):
    if specific_anchor == 'default':
        block = nn.Conv2D(out_channel, kernel_size=1, padding=0, strides=1)
    elif specific_anchor == 'square':
        block = YOLOOutputV3SquareKernel(sa_level, kernel, out_channel, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
    else:
        block = YOLOOutputV3RectangleKernel(sa_level, kernel, out_channel, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
    return block


class Pad(gluon.HybridBlock):

    def __init__(self, u, d, l, r, **kwargs):
        super(Pad, self).__init__(**kwargs)
        self.u, self.d, self.l, self.r = u, d, l, r

    def hybrid_forward(self, F, x):
        return F.pad(x, mode="constant", pad_width=(0, 0, 0, 0, self.u, self.d, self.l, self.r))


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
    ignore_iou_thresh : float
        ignore the box loss whose iou with the gt larger than the ignore_iou_thresh
        current sigmoid config
    alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, Scalar, etc.
    """

    def __init__(self, index, num_class, anchors, stride, ignore_iou_thresh, coop_config=None, alloc_size=(128, 128),
                 specific_anchor='default', sa_level=1, kernels=None, coop_loss=False, norm_layer=BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(YOLOOutputV3, self).__init__(**kwargs)
        anchors = np.array(anchors).astype('float32')
        # suppose we don't config the xywho numbers more than or equal with 3
        assert len(coop_config) % 3 == 0
        self._xywho_num = len(coop_config) // 3
        self._classes = num_class

        self._coop_loss = coop_loss
        self._num_pred = self._xywho_num * (1 + 4) + num_class
        self._num_anchors = anchors.size // 2
        self.all_pred = self._num_pred * self._num_anchors

        self._stride = stride
        dic = {32: slice(0, 1), 16: slice(1, 5), 8: slice(5, 21)}
        self._loss = SelfLoss(index, self._classes, ignore_iou_thresh, coop_config, dic[self._stride])
        if self._coop_loss:
            self._cooploss = SelfCoopLoss(self._classes, coop_config, dic[self._stride])
        with self.name_scope():
            self.prediction = select_yolo_output(self.all_pred, specific_anchor, sa_level, kernels,
                                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            # nn.Conv2D(self.all_pred, kernel_size=1, padding=0, strides=1)
            # anchors will be multiplied to predictions
            anchors = anchors.reshape((1, 1, -1, 1, 2))
            self.anchors_self = self.params.get_constant('anchor_%d' % (index), anchors)
            # offsets will be added to predictions
            grid_x = np.arange(alloc_size[1])
            grid_y = np.arange(alloc_size[0])
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            # stack to (n, n, 2)
            offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)
            # expand dims to (1, 1, n, n, 2) so it's easier for broadcasting
            offsets = np.expand_dims(np.expand_dims(offsets, axis=0), axis=0)
            self.offsets_self = self.params.get_constant('offset_%d' % (index), offsets)
            horizontal_sig_levels = np.tile(np.array(coop_config)[np.newaxis, np.newaxis, :, :, np.newaxis],
                                            reps=(alloc_size[0], alloc_size[1], 1, 1, 1))
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

    def hybrid_forward(self, F, x, *args, anchors_self, offsets_self, horizontal_sig_levels):
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
        pred = self.prediction(x).reshape((0, self.all_pred, -1)).transpose(axes=(0, 2, 1))
        # transpose to (batch, height * width, num_anchor, num_pred)
        pred = pred.reshape((0, -1, self._num_anchors, self._num_pred))

        # components
        # transpose to (batch, height * width, num_anchor, num_xywho, 5)
        xywho = pred.slice_axis(axis=-1, begin=0, end=-self._classes).reshape((0, 0, 0, self._xywho_num, 5))
        class_pred = pred.slice_axis(axis=-1, begin=-self._classes, end=None)
        ctrs = xywho.slice_axis(axis=-1, begin=0, end=2)
        raw_box_scales = xywho.slice_axis(axis=-1, begin=2, end=4)
        objness = xywho.slice_axis(axis=-1, begin=4, end=None)

        # valid offsets, (1, 1, height, width, 2)
        align_x = F.slice_axis(F.slice_axis(x, axis=0, begin=0, end=1), axis=1, begin=0, end=3).transpose(axes=(0, 2, 3, 1))
        offsets = F.slice_like(offsets_self, x * 0, axes=(2, 3))
        horizontal_sig_levels = F.slice_like(horizontal_sig_levels, F.transpose(x * 0, axes=(2, 3, 0, 1)),
                                             axes=(0, 1)).reshape((-3, 0, 0, 0)).expand_dims(axis=0)
        # reshape to (1, height*width, 1, 2)
        offsets = F.broadcast_sub(offsets.reshape((1, -1, 1, 2)).expand_dims(-2) + 0.5, 0.5 * horizontal_sig_levels)

        box_centers = F.broadcast_add(F.broadcast_mul(F.sigmoid(ctrs), horizontal_sig_levels), offsets) * self._stride
        box_scales = F.broadcast_mul(F.exp(raw_box_scales), anchors_self)
        confidence = F.sigmoid(xywho.slice_axis(axis=-1, begin=4, end=5))
        class_score = F.broadcast_mul(F.sigmoid(class_pred).expand_dims(-2), confidence)
        wh = box_scales / 2.0
        bbox = F.concat(box_centers - wh, box_centers + wh, dim=-1)

        if autograd.is_training():
            # during training, we don't need to convert whole bunch of info to detection results
            if autograd.is_recording():
                objness = objness.reshape((0, -3, 0, -1))
                ctrs = ctrs.reshape((0, -3, 0, 2))
                raw_box_scales = raw_box_scales.reshape((0, -3, 0, 2))
                class_pred = class_pred.reshape((0, -3, -1))
                # bbox = bbox.reshape((0, -1, 4))
                if self._coop_loss:
                    return self._loss(objness, ctrs, raw_box_scales, class_pred, bbox.reshape((0, -1, 4)), horizontal_sig_levels.reshape((0, -3, -1, 1)), *args[:-2]) + \
                           self._cooploss(ctrs, raw_box_scales, align_x, horizontal_sig_levels.reshape((0, -3, -1, 1)), *args[-3:])
                else:
                    return self._loss(objness, ctrs, raw_box_scales, class_pred, bbox.reshape((0, -1, 4)), horizontal_sig_levels.reshape((0, -3, -1, 1)), *args)
            else:
                return anchors_self, offsets.slice(begin=(None, None, 0, 0, 0), end=(None, None, 1, 1, 1))

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


class YOLOOutputV3SquareKernel(gluon.HybridBlock):
    """YOLO V3 Detection Block which does the following:
    - add a few conv layers
    - return the output
    - have a branch that do yolo detection.
    Parameters
    ----------
    channel : int
        Number of channels for 1x1 conv. 3x3 Conv will have 2*channel.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, sa_level, kernel, out_channel, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(YOLOOutputV3SquareKernel, self).__init__(**kwargs)
        assert out_channel % 3 == 0, "channel {} cannot be divided by 3".format(out_channel)
        self.sa_level = sa_level
        self.kernel = kernel
        self.anchor_channel = out_channel // 3
        with self.name_scope():
            self.conv0 = _conv2d(out_channel, 1, 0, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            for i in range(3):
                for ii in range(sa_level):
                    setattr(self, 'anchor' + str(i), nn.HybridSequential(prefix=''))
                    getattr(self, 'anchor' + str(i)).add(_conv2d(self.anchor_channel * 2, kernel, (kernel-1)//2, 1,
                                                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                    if ii == sa_level - 1:
                        conv_1x1 = nn.Conv2D(self.anchor_channel, kernel_size=1, padding=0, strides=1)
                    else:
                        conv_1x1 = _conv2d(self.anchor_channel, 1, 0, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                    getattr(self, 'anchor' + str(i)).add(conv_1x1)

    def hybrid_forward(self, F, x):
        x = self.conv0(x)
        anchor0 = self.anchor0(x)
        anchor1 = self.anchor1(x)
        anchor2 = self.anchor2(x)
        return F.concat(anchor0, anchor1, anchor2, dim=1)


class YOLOOutputV3RectangleKernel(gluon.HybridBlock):
    """YOLO V3 Detection Block which does the following:
    - add a few conv layers
    - return the output
    - have a branch that do yolo detection.
    Parameters
    ----------
    channel : int
        Number of channels for 1x1 conv. 3x3 Conv will have 2*channel.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, sa_level, kernel, out_channel, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(YOLOOutputV3RectangleKernel, self).__init__(**kwargs)
        assert out_channel % 3 == 0, "channel {} cannot be divided by 3".format(out_channel)

        self.sa_level = sa_level
        self.kernel = kernel
        self.pad_np = np.array(kernel) - 1  # shape(3, 2)
        self.pad_ul = self.pad_np // 2
        self.pad_dr = self.pad_np - self.pad_ul
        self.anchor_channel = out_channel // 3
        with self.name_scope():
            self.conv0 = _conv2d(out_channel, 1, 0, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            for i in range(3):
                for ii in range(sa_level):
                    setattr(self, 'anchor' + str(i), nn.HybridSequential(prefix=''))
                    getattr(self, 'anchor' + str(i)).add(Pad(self.pad_ul[i, 0], self.pad_dr[i, 0],
                                                             self.pad_ul[i, 1], self.pad_dr[i, 1]))
                    getattr(self, 'anchor' + str(i)).add(_conv2d(self.anchor_channel * 2, kernel[i], 0, 1,
                                                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                    if ii == sa_level-1:
                        conv_1x1 = nn.Conv2D(self.anchor_channel, kernel_size=1, padding=0, strides=1)
                    else:
                        conv_1x1 = _conv2d(self.anchor_channel, 1, 0, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                    getattr(self, 'anchor' + str(i)).add(conv_1x1)

    def hybrid_forward(self, F, x):
        x = self.conv0(x)
        anchor0 = self.anchor0(x)
        anchor1 = self.anchor1(x)
        anchor2 = self.anchor2(x)
        return F.concat(anchor0, anchor1, anchor2, dim=1)


