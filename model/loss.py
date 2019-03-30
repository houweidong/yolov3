from mxnet import gluon
# from mxnet import nd
from mxnet import autograd
from mxnet.gluon.loss import Loss
from model.target import SelfDynamicTargetGeneratorSimple, get_factor


class SelfLoss(Loss):
    """Losses of YOLO v3.

    Parameters
    ----------
    num_class : int
    ignore_iou_thresh : float
        ignore the box loss whose iou with the gt larger than the ignore_iou_thresh
    coop_configs : tuple, such as (1, )
        current sigmoid config
    target_slice : slice
        slice the target to take the target belongs to current output layer
    label_smooth : bool
        Whether open label smooth in the train phase
    batch_axis : int, default 0
        The axis that represents mini-batch.
    weight : float or None
        Global scalar weight for loss.
    coop_mode : string
        "flat", different level grids have same weight loss in the training phase
        "convex", the center grids have higher weight than the marginal grids in the training phase
        "concave", the marginal grids have higher weight than the center grids in the training phase
        "equal", consider the num of the same level grids to make loss equal
    sigma_weight : float
        for coop_mode params, we use Gaussian distribution to generate the weights according to the grid level,
        the sigma_weight is the distribution's params

    """

    def __init__(self, num_class, ignore_iou_thresh, coop_configs, target_slice, label_smooth,
                 coop_mode, sigma_weight, batch_axis=0, weight=None, **kwargs):
        super(SelfLoss, self).__init__(weight, batch_axis, **kwargs)
        self._coop_mode = coop_mode
        self._sigma_weight = sigma_weight
        self._target_slice = target_slice
        self._num_class = num_class
        self._sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        # self._ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
        self._l1_loss = gluon.loss.L1Loss()
        # self._l2_loss = gluon.loss.L2Loss()
        self._label_smooth = label_smooth
        self._coop_configs = coop_configs
        self.order_sig_config = sorted(set(coop_configs))
        self._dynamic_target = SelfDynamicTargetGeneratorSimple(num_class, ignore_iou_thresh)
        self._factor_list, self._center_list, self._factor_max, self._center_max = \
            get_factor(coop_configs, coop_mode, sigma_weight)

    def hybrid_forward(self, F, objness, box_centers, box_scales, cls_preds, box_preds, gt_boxes,
                       objness_t, center_t, scale_t, weight_t, cls_mask, obj_mask, box_index):
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
        box_index : mxnet.nd.NDArray
            range (0, n), indicate the grid belongs to which ground truth box in gt_boxes
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
        batch_ious, dynamic_objness = self._dynamic_target(box_preds, gt_boxes)
        # if len(self._coop_configs) == 1:
        #     ious_max, dynamic_objness, objness, box_centers, box_scales = \
        #         [[ious_max], [dynamic_objness], [objness], [box_centers], [box_scales]]
        # else:
        #     box_centers, box_scales = \
        #         [[pp.reshape((0, -1, 2)) for pp in p.reshape((0, -1, len(self._coop_configs), 2)).split(
        #             num_outputs=len(self._coop_configs), axis=2)] for p in [box_centers, box_scales]]
        if len(self._coop_configs) != 1:
            batch_ious, dynamic_objness, objness, box_centers, box_scales = \
                [[pp.reshape((0, -3, -1)) for pp in p.reshape((0, -4, -1, len(self._coop_configs), 0)).split(
                    num_outputs=len(self._coop_configs), axis=2)] for p in
                 [batch_ious, dynamic_objness, objness, box_centers, box_scales]]
        else:
            batch_ious, dynamic_objness, objness, box_centers, box_scales = \
                [batch_ious], [dynamic_objness], [objness], [box_centers], [box_scales]

        objness_t, center_t, scale_t, weight_t, cls_mask, obj_mask, box_index = \
            [F.concat(*(p.split(num_outputs=21, axis=1)[self._target_slice]), dim=1) for p in
             [objness_t, center_t, scale_t, weight_t, cls_mask, obj_mask, box_index]]

        # the coop_config is ordered in main, but one value can appear more than one time,
        # so when current level value is same as the previous', the index will not increment
        level_old, level_index = 0, -3
        for index_xywho, level in enumerate(self._coop_configs):
            with autograd.pause():
                denorm = F.cast(F.shape_array(objness_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
                mask = obj_mask <= level

                if self._coop_mode == 'flat':
                    grid_weight = self._factor_list[index_xywho]
                elif self._coop_mode in ['convex', 'concave']:
                    grid_weight = F.exp(-1 * F.square(obj_mask - self._center_list[index_xywho])
                                        / (self._sigma_weight ** 2)) * self._factor_list[index_xywho]
                elif self._coop_mode == 'equal':
                    grid_weight = (1 / (2*obj_mask-1)) * self._factor_list[index_xywho]
                else:
                    raise Exception('coop_mode error in loss layer when compute cls loss')

                obj = F.where(mask, objness_t * grid_weight, dynamic_objness[index_xywho])
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
                hard_objness_fit = F.pick(batch_ious[index_xywho], index=box_index.squeeze(axis=-1), axis=-1, keepdims=True)
                # recover hard_objness_t with iou, if box_index has been valued with the box id
                hard_objness_t = F.where(F.where(mask, box_index, -F.ones_like(mask)) == -1, hard_objness_t, hard_objness_fit)
                new_objness_mask = F.where(obj > 0, obj, obj >= 0)
            obj_loss = F.broadcast_mul(self._sigmoid_ce(objness[index_xywho], hard_objness_t, new_objness_mask), denorm)
            # level ** 0.5 for incrementing the loss for high sigmoid level
            center_loss = (level ** 0.5) * F.broadcast_mul(self._sigmoid_ce(box_centers[index_xywho], ctr, weight), denorm * 2)
            scale_loss = F.broadcast_mul(self._l1_loss(box_scales[index_xywho], scl, weight), denorm * 2)
            if level != level_old:
                level_index += 3
            loss[level_index] = obj_loss + loss[level_index]
            loss[level_index + 1] = center_loss + loss[level_index + 1]
            loss[level_index + 2] = scale_loss + loss[level_index + 2]
            level_old = level
        with autograd.pause():
            mask3 = cls_mask <= max(self._coop_configs)
            mask4 = F.max(mask3, axis=-1, keepdims=True).tile(reps=(self._num_class,))
            cls = F.where(mask3, F.ones_like(mask3), F.zeros_like(mask3))
            smooth_weight = 1. / self._num_class
            if self._label_smooth:
                smooth_weight = 1. / self._num_class
                cls = F.where(cls > 0.5, cls - smooth_weight, F.ones_like(cls) * smooth_weight)
            denorm_class = F.cast(F.shape_array(cls).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')

            if self._coop_mode == 'flat':
                grid_weight = self._factor_max
            elif self._coop_mode in ['convex', 'concave']:
                grid_weight = F.exp(-1*F.square(obj_mask-self._center_max)/(self._sigma_weight**2)) * self._factor_max
            elif self._coop_mode == 'equal':
                grid_weight = (1 / (2 * obj_mask - 1)) * self._factor_max
            else:
                raise Exception('coop_mode error in loss layer when compute cls loss')

            class_mask = F.broadcast_mul(mask4, objness_t * grid_weight)
        cls_loss = F.broadcast_mul(self._sigmoid_ce(cls_preds, cls, class_mask), denorm_class)
        loss[-1] = cls_loss + loss[-1]
        return loss
