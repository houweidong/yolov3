from mxnet import gluon
# from mxnet import nd
from mxnet import autograd
from mxnet.gluon.loss import Loss  # , _apply_weighting, _reshape_like
from self_target import SelfDynamicTargetGeneratorSimple
# from collections import OrderedDict
# from utils import get_order_config


class SelfLoss(Loss):
    """Losses of YOLO v3.

    Parameters
    ----------
    batch_axis : int, default 0
        The axis that represents mini-batch.
    weight : float or None
        Global scalar weight for loss.

    """

    def __init__(self, num_class, ignore_iou_thresh, coop_configs, target_slice, label_smooth,
                 batch_axis=0, weight=None, **kwargs):
        super(SelfLoss, self).__init__(weight, batch_axis, **kwargs)
        self._target_slice = target_slice
        self._num_class = num_class
        self._sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        # self._ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
        self._l1_loss = gluon.loss.L1Loss()
        self._l2_loss = gluon.loss.L2Loss()
        self._dynamic_target = SelfDynamicTargetGeneratorSimple(num_class, ignore_iou_thresh)
        self._label_smooth = label_smooth
        self._coop_configs = coop_configs
        self.order_sig_config = sorted(set(coop_configs))

    def hybrid_forward(self, F, objness, box_centers, box_scales, cls_preds, box_preds, gt_boxes,
                       objness_t, center_t, scale_t, weight_t, cls_mask, obj_mask):
        """Compute YOLOv3 losses.

        Parameters
        ----------
        objness : mxnet.nd.NDArray
            Predicted objectness (B, N), range (0, 1).
        box_centers : mxnet.nd.NDArray
            Predicted box centers (x, y) (B, N, 2), range (0, 1).
        box_scales : mxnet.nd.NDArray
            Predicted box scales (width, height) (B, N, 2).
        cls_preds : mxnet.nd.NDArray
            Predicted class predictions (B, N, num_class), range (0, 1).
        box_preds : mxnet.nd.NDArray
            Predicted bounding boxes.
        gt_boxes : mxnet.nd.NDArray
            Ground-truth bounding boxes.
        objness_t : mxnet.nd.NDArray
            Prefetched Objectness targets.
        center_t : mxnet.nd.NDArray
            Prefetched regression target for center x and y.
        scale_t : mxnet.nd.NDArray
            Prefetched regression target for scale x and y.
        weight_t : mxnet.nd.NDArray
            Prefetched element-wise gradient weights for center_targets and scale_targets.
        cls_mask : mxnet.nd.NDArray
            range (0, 1000), include xywho level information
        obj_mask : mxnet.nd.NDArray
            range (0, 100), include xywho level information
        objectness_cls : mxnet.nd.NDArray
            range (0, 1)
        Returns
        -------
        tuple of NDArrays
            obj_loss: sum of objectness logistic loss
            center_loss: sum of box center logistic regression loss
            scale_loss: sum of box scale l1 loss
            cls_loss: sum of per class logistic loss

        """

        loss = []
        for _ in self.order_sig_config:
            loss.append(0.)
            loss.append(0.)
            loss.append(0.)
        loss.append(0.)
        dynamic_objness = self._dynamic_target(box_preds, gt_boxes)
        if len(self._coop_configs) == 1:
            dynamic_objness, objness, box_centers, box_scales = \
                [[dynamic_objness], [objness], [box_centers], [box_scales]]
        else:
            dynamic_objness, objness = \
                [p.reshape(0, -1, len(self._coop_configs), 1).split(num_outputs=len(self._coop_configs), axis=2)
                     .reshape(0, -1, 1) for p in[dynamic_objness, objness]]
            box_centers, box_scales = \
                [p.reshape(0, -1, len(self._coop_configs), 2).split(num_outputs=len(self._coop_configs), axis=2)
                     .reshape(0, -1, 2) for p in [box_centers, box_scales]]

        objness_t, center_t, scale_t, weight_t, cls_mask, obj_mask = \
            [F.concat(*(p.split(num_outputs=21, axis=1)[self._target_slice]), dim=1) for p in
             [objness_t, center_t, scale_t, weight_t, cls_mask, obj_mask]]

        # the coop_config is ordered in main, but one value can appear more than one time,
        # so when current level value is same as the previous', the index will not increment
        level_old, level_index = 0, -3
        denorm = F.cast(F.shape_array(objness_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        for index_xywho in range(len(self._coop_configs)):
            with autograd.pause():
                level = self._coop_configs[index_xywho]
                mask = obj_mask <= level
                obj = F.where(mask, objness_t, dynamic_objness[index_xywho])
                mask2 = mask.tile(reps=(2,))
                # + 0.5 level to transform the range(-0.5*level, 0.5*level) to range(0, level)
                ctr = F.where(mask2, (center_t + 0.5 * level) / float(level), F.zeros_like(mask2))
                # similar to label smooth, here smooth the 0 and 1 label for x y
                # ctr = F.where(ctr>=0.95, F.ones_like(ctr)*0.95, ctr)
                # ctr = F.where(ctr<=0.05, F.ones_like(ctr)*0.05, ctr)
                scl = F.where(mask2, scale_t, F.zeros_like(mask2))
                wgt = F.where(mask2, weight_t, F.zeros_like(mask2))

                weight = F.broadcast_mul(wgt, obj)
                hard_objness_t = F.where(obj > 0, F.ones_like(obj), obj)
                new_objness_mask = F.where(obj > 0, obj, obj >= 0)
            obj_loss = F.broadcast_mul(self._sigmoid_ce(objness[index_xywho], hard_objness_t, new_objness_mask), denorm)
            center_loss = F.broadcast_mul(self._sigmoid_ce(box_centers[index_xywho], ctr, weight), denorm * 2)
            scale_loss = F.broadcast_mul(self._l1_loss(box_scales[index_xywho], scl, weight), denorm * 2)
            if level != level_old:
                level_index += 3
            loss[level_index] = obj_loss + loss[level_index]
            loss[level_index + 1] = center_loss + loss[level_index + 1]
            loss[level_index + 2] = scale_loss + loss[level_index + 2]
        with autograd.pause():
            mask3 = cls_mask <= max(self._coop_configs)
            mask4 = F.max(mask3, axis=-1, keepdims=True).tile(reps=(self._num_class,))
            cls = F.where(mask3, F.ones_like(mask3), F.zeros_like(mask3))
            smooth_weight = 1. / self._num_class
            if self._label_smooth:
                smooth_weight = 1. / self._num_class
                cls = F.where(cls > 0.5, cls - smooth_weight, F.ones_like(cls) * smooth_weight)
            denorm_class = F.cast(F.shape_array(cls).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
            class_mask = F.broadcast_mul(mask4, objness_t)
        cls_loss = F.broadcast_mul(self._sigmoid_ce(cls_preds, cls, class_mask), denorm_class)
        loss[-1] = cls_loss + loss[-1]
        return loss

