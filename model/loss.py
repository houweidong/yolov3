from mxnet import gluon
# from mxnet import nd
from mxnet import autograd
from mxnet.gluon.loss import Loss
from model.target import SelfDynamicTargetGeneratorSimple


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

    def __init__(self, index, num_class, ignore_iou_thresh, coop_configs, target_slice, batch_axis=0, weight=None, **kwargs):
        super(SelfLoss, self).__init__(weight, batch_axis, **kwargs)
        self._target_slice = target_slice
        self._num_class = num_class
        self._sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self._l1_loss = gluon.loss.L1Loss()
        # self._l2_loss = gluon.loss.L2Loss()
        self._coop_configs = coop_configs
        self._dynamic_target = SelfDynamicTargetGeneratorSimple(num_class, ignore_iou_thresh, coop_configs.shape[-1])

    def hybrid_forward(self, F, objness, box_centers, box_scales, cls_preds, box_preds, coop, gt_boxes,
                       center_t, scale_t, box_index, obj_t, cls_mask, cls_fct, weight_t):
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
        weights_balance : mxnet.nd.NDArray
            to equal train every target
        Returns
        -------
        tuple of NDArrays
            obj_loss: sum of objectness logistic loss
            center_loss: sum of box center logistic regression loss
            scale_loss: sum of box scale l1 loss
            cls_loss: sum of per class logistic loss

        """
        # cls_mask = F.concat(*(cls_mask.split(num_outputs=21, axis=1)[self._target_slice]), dim=1)
        # if self._coop_configs.shape[-1] != 1:
        #     batch_ious, dynamic_objness, objness, box_centers, box_scales, weights_balance = \
        #         ([pp.reshape((0, -3, -1)) for pp in p.reshape((0, -4, -1, self._coop_configs.shape[-1], 0)).split(
        #             num_outputs=self._coop_configs.shape[-1], axis=2)] for p in
        #          [batch_ious, dynamic_objness, objness, box_centers, box_scales, weights_balance])
        # else:
        #     batch_ious, dynamic_objness, objness, box_centers, box_scales, weights_balance = \
        #         [batch_ious], [dynamic_objness], [objness], [box_centers], [box_scales], [weights_balance]

        # the coop_config is ordered in main, but one value can appear more than one time,
        # so when current level value is same as the previous', the index will not increment
        # level_old, level_index = 0, -3
        # for index_xywho, level in enumerate(self._coop_configs):
        #     with autograd.pause():
        #         denorm = F.cast(F.shape_array(objness_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        #         mask = obj_mask <= level
        #
        #         if self._coop_mode == 'flat':
        #             grid_weight = self._factor_list[index_xywho]
        #         elif self._coop_mode in ['convex', 'concave']:
        #             grid_weight = F.exp(-1 * F.square(obj_mask - self._center_list[index_xywho])
        #                                 / (self._sigma_weight ** 2)) * self._factor_list[index_xywho]
        #         # TODO  to omit this option because the weights_balance is inconsistent with it
        #         elif self._coop_mode == 'equal':
        #             grid_weight = (1 / (2*obj_mask-1)) * self._factor_list[index_xywho]
        #         else:
        #             raise Exception('coop_mode error in loss layer when compute cls loss')
        #
        #         # obj just a weight integration(mixup weight, equal train weight, and grid weight(flat convex concave))
        #         obj = F.where(mask, objness_t * grid_weight * weights_balance[index_xywho], dynamic_objness[index_xywho])
        #         # mask2 =
        #         # ctr = F.where(mask2, (center_t + 0.5 * level) / float(level), F.zeros_like(mask2))
        #         # similar to label smooth, here smooth the 0 and 1 label for x y
        #         # ctr = F.where(ctr>=0.95, F.ones_like(ctr)*0.95, ctr)
        #         # ctr = F.where(ctr<=0.05, F.ones_like(ctr)*0.05, ctr)
        #         # scl = F.where(mask2, scale_t, F.zeros_like(mask2))
        #         # weight just a weight integration(wh weight, mixup weight, equal train weight)
        #         weight = F.broadcast_mul(F.where(mask.tile(reps=(2,)), weight_t, F.zeros_like(weight_t)), obj)
        #         hard_objness_t = F.where(obj > 0, F.ones_like(obj), obj)
        #         hard_objness_fit = F.pick(batch_ious[index_xywho], index=box_index.squeeze(axis=-1), axis=-1, keepdims=True)
        #         # recover hard_objness_t with iou, if box_index has been valued with the box id
        #         hard_objness_t = F.where(F.where(mask, box_index, -F.ones_like(mask)) == -1, hard_objness_t, hard_objness_fit)
        #         new_objness_mask = F.where(obj > 0, obj, obj >= 0)
        with autograd.pause():
            batch_ious, dynamic_objness = self._dynamic_target(box_preds, gt_boxes)
            center_t, scale_t, box_index = (F.tile(F.concat(*(p.split(num_outputs=21, axis=1)[self._target_slice]),
                dim=1), reps=(self._coop_configs.shape[-1], 1)) for p in [center_t, scale_t, box_index])
            obj_t, weight_t, cls_mask, cls_fct = (F.concat(*(p.split(num_outputs=21, axis=1)[self._target_slice]),
                dim=1) for p in [obj_t, weight_t, cls_mask, cls_fct])
            denorm = F.cast(F.shape_array(obj_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
            # just to multi weights_balance, objness_t, weight_t together
            # obj_wb_fc = F.broadcast_mul(objness_t, weights_balance)
            # weight = F.broadcast_mul(obj_wb_fc, weight_t)
            # if self._separate:
            #     mask_obj = (cls_mask >= 0).tile(reps=(self._coop_configs.shape[-1], 1))
            # else:
            #     mask_obj = weights_balance != 0

            # suppose that coord area is larger than the obj area
            obj = F.where(obj_t > 0, obj_t, dynamic_objness)
            hard_objness_t = F.where(obj > 0, F.ones_like(obj), obj)
            hard_objness_fit = F.pick(batch_ious, index=box_index.squeeze(axis=-1), axis=-1, keepdims=True)
            # recover hard_objness_t with iou, if box_index has been valued with the box id
            hard_objness_t = F.where(F.where(obj_t > 0, box_index, -F.ones_like(obj_t)) == -1, hard_objness_t, hard_objness_fit)
            new_objness_mask = F.where(obj > 0, obj, obj >= 0)

        obj_loss = F.broadcast_mul(self._sigmoid_ce(objness, hard_objness_t, new_objness_mask), denorm)
        center_loss = F.broadcast_mul(self._sigmoid_ce(box_centers, F.broadcast_div(F.broadcast_add(
            center_t, 0.5 * coop), coop), F.broadcast_mul(weight_t, coop**0.3)), denorm*2)
        # center_loss = F.broadcast_mul(self._sigmoid_ce(box_centers, F.broadcast_div(F.broadcast_add(
        #     center_t, 0.5 * coop), coop), weight), denorm*2)
        scale_loss = F.broadcast_mul(self._l1_loss(box_scales, scale_t, weight_t), denorm * 2)

        with autograd.pause():
            # if self._coop_configs.shape[-1] != 1:
            #     objness_t = objness_t.split(num_outputs=self._coop_configs.shape[-1], axis=2)[0]
            # mask3 = cls_mask >= 0
            # cls = F.one_hot(cls_mask.squeeze(axis=-1), depth=self._num_class)
            # if self._label_smooth:
            #     smooth_weight = 1. / self._num_class
            #     cls = F.where(cls > 0.5, cls - smooth_weight, F.ones_like(cls) * smooth_weight)
            # class_mask = F.broadcast_mul(mask3.tile(reps=(self._num_class,)), objness_t)
            denorm_class = F.cast(F.shape_array(cls_mask).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        cls_loss = F.broadcast_mul(self._sigmoid_ce(cls_preds.expand_dims(-2), cls_mask, cls_fct.tile(reps=(self._num_class, ))), denorm_class)

        return obj_loss, center_loss, scale_loss, cls_loss


class SelfCoopLoss(Loss):

    def __init__(self, num_class, coop_configs, target_slice, batch_axis=0, weight=None, **kwargs):
        super(SelfCoopLoss, self).__init__(weight, batch_axis, **kwargs)
        self._target_slice = target_slice
        self._num_class = num_class
        self._sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self._l1_loss = gluon.loss.L1Loss()
        self._coop_configs = coop_configs

    def hybrid_forward(self, F, box_centers, box_scales, align_x, coop, weight_t, mask_vtc, mask_hrz):

        with autograd.pause():
            weight_t, mask_vtc, mask_hrz = (F.concat(*(p.split(num_outputs=21, axis=1)[self._target_slice]), dim=1)
                                            for p in [mask_vtc, mask_hrz, weight_t])
            denorm = F.cast(F.shape_array(weight_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
            mask_bvtc = mask_vtc.reshape_like(rhs=align_x, lhs_begin=1, lhs_end=2, rhs_begin=1, rhs_end=None).tile(reps=(2, ))
            mask_bhrz = mask_hrz.reshape_like(rhs=align_x, lhs_begin=1, lhs_end=2, rhs_begin=1, rhs_end=None).tile(reps=(2, ))
            coop = coop.reshape_like(rhs=align_x, lhs_begin=1, lhs_end=2, rhs_begin=1, rhs_end=None)

        # with autograd.pause():
        bct_rs = box_centers.reshape_like(rhs=align_x, lhs_begin=1, lhs_end=2, rhs_begin=1, rhs_end=None)
        bct_td = F.concat(bct_rs.slice_axis(axis=1, begin=1, end=None), bct_rs.slice_axis(axis=1, begin=0, end=1), dim=1)
        bct_lr = F.concat(bct_rs.slice_axis(axis=2, begin=1, end=None), bct_rs.slice_axis(axis=2, begin=0, end=1), dim=2)

        with autograd.pause():
            bct_rs_sigmoid = F.sigmoid(bct_rs)
            bct_td_sigmoid = F.sigmoid(bct_td)
            bct_lr_sigmoid = F.sigmoid(bct_lr)
            bct_td_target1 = F.clip(F.broadcast_div(F.broadcast_mul(bct_td_sigmoid.slice_axis(axis=-1, begin=-1, end=None), coop) + 1, coop), a_min=0., a_max=1.)
            bct_td_target1 = F.concat(bct_td_sigmoid.slice_axis(axis=-1, begin=0, end=1), bct_td_target1, dim=-1)
            bct_td_target2 = F.clip(F.broadcast_div(F.broadcast_mul(bct_rs_sigmoid.slice_axis(axis=-1, begin=-1, end=None), coop) - 1, coop), a_min=0., a_max=1.)
            bct_td_target2 = F.concat(bct_rs_sigmoid.slice_axis(axis=-1, begin=0, end=1), bct_td_target2, dim=-1)
            bct_td_target = F.concat(bct_td_target1, bct_td_target2, dim=-1)

            bct_lr_target1 = F.clip(F.broadcast_div(F.broadcast_mul(bct_lr_sigmoid.slice_axis(axis=-1, begin=0, end=1), coop) + 1, coop), a_min=0., a_max=1.)
            bct_lr_target1 = F.concat(bct_lr_target1, bct_lr_sigmoid.slice_axis(axis=-1, begin=-1, end=None), dim=-1)
            bct_lr_target2 = F.clip(F.broadcast_div(F.broadcast_mul(bct_rs_sigmoid.slice_axis(axis=-1, begin=0, end=1), coop) - 1, coop), a_min=0., a_max=1.)
            bct_lr_target2 = F.concat(bct_lr_target2, bct_rs_sigmoid.slice_axis(axis=-1, begin=-1, end=None), dim=-1)
            bct_lr_target = F.concat(bct_lr_target1, bct_lr_target2, dim=-1)
        bct_coloss = F.broadcast_mul(self._sigmoid_ce(F.concat(bct_rs, bct_td, dim=-1), bct_td_target, mask_bvtc), denorm)
        bct_coloss = bct_coloss + F.broadcast_mul(self._sigmoid_ce(F.concat(bct_rs, bct_lr, dim=-1), bct_lr_target, mask_bhrz), denorm)

        bsc_rs = box_scales.reshape_like(rhs=align_x, lhs_begin=1, lhs_end=2, rhs_begin=1, rhs_end=None)
        bsc_tb = F.concat(bsc_rs.slice_axis(axis=1, begin=1, end=None), bsc_rs.slice_axis(axis=1, begin=0, end=1), dim=1)
        bsc_lr = F.concat(bsc_rs.slice_axis(axis=2, begin=1, end=None), bsc_rs.slice_axis(axis=2, begin=0, end=1), dim=2)

        with autograd.pause():
            bsc_td_target = F.concat(bsc_tb, bsc_rs, dim=-1)
            bsc_lr_target = F.concat(bsc_lr, bsc_rs, dim=-1)
        bsc_coloss = F.broadcast_mul(self._l1_loss(F.concat(bsc_rs, bsc_tb, dim=-1), bsc_td_target, mask_bvtc), denorm)
        bsc_coloss = bsc_coloss + F.broadcast_mul(self._l1_loss(F.concat(bsc_rs, bsc_lr, dim=-1), bsc_lr_target, mask_bhrz), denorm)

        # obj_rs = objness.reshape_like(rhs=align_x, lhs_begin=1, lhs_end=2, rhs_begin=1, rhs_end=None)
        # obj_tb = F.concat(obj_rs.slice_axis(axis=1, begin=1, end=None), obj_rs.slice_axis(axis=1, begin=0, end=1), dim=1)
        # obj_lr = F.concat(obj_rs.slice_axis(axis=2, begin=1, end=None), obj_rs.slice_axis(axis=2, begin=0, end=1), dim=2)

        # with autograd.pause():
        #     mask_ojcl_vtc = F.broadcast_mul(mask_obj, mask_vtc).reshape_like(rhs=align_x, lhs_begin=1, lhs_end=2, rhs_begin=1, rhs_end=None)
        #     mask_ojcl_hrz = F.broadcast_mul(mask_obj, mask_hrz).reshape_like(rhs=align_x, lhs_begin=1, lhs_end=2, rhs_begin=1, rhs_end=None)
        #     obj_td_target = F.sigmoid(F.concat(obj_tb, obj_rs, dim=-1))
        #     obj_lr_target = F.sigmoid(F.concat(obj_lr, obj_rs, dim=-1))
        # obj_coloss = F.broadcast_mul(self._sigmoid_ce(
        #     F.concat(obj_rs, obj_tb, dim=-1), obj_td_target, mask_ojcl_vtc.tile(reps=(2,))), denorm*2)
        # obj_coloss = obj_coloss + F.broadcast_mul(self._sigmoid_ce(
        #     F.concat(obj_rs, obj_lr, dim=-1), obj_lr_target, mask_ojcl_hrz.tile(reps=(2,))), denorm*2)
        #
        # denorm_cls = denorm/self._coop_configs.shape[-1]*self._num_class
        # cls_rs = cls_preds.expand_dims(-2).reshape_like(rhs=align_x, lhs_begin=1, lhs_end=2, rhs_begin=1, rhs_end=None)
        # cls_tb = F.concat(cls_rs.slice_axis(axis=1, begin=1, end=None), cls_rs.slice_axis(axis=1, begin=0, end=1), dim=1)
        # cls_lr = F.concat(cls_rs.slice_axis(axis=2, begin=1, end=None), cls_rs.slice_axis(axis=2, begin=0, end=1), dim=2)
        #
        # with autograd.pause():
        #     cls_td_target = F.sigmoid(F.concat(cls_tb, cls_rs, dim=-1))
        #     cls_lr_target = F.sigmoid(F.concat(cls_lr, cls_rs, dim=-1))
        # cls_coloss = F.broadcast_mul(self._sigmoid_ce(F.concat(cls_rs, cls_tb, dim=-1),
        #     cls_td_target, mask_ojcl_vtc.tile(reps=(2*self._num_class, ))), denorm_cls * 2)
        # cls_coloss = cls_coloss + F.broadcast_mul(self._sigmoid_ce(F.concat(cls_rs, cls_lr, dim=-1),
        #     cls_lr_target, mask_ojcl_vtc.tile(reps=(2*self._num_class,))), denorm_cls * 2)

        # return obj_coloss, bct_coloss, bsc_coloss, cls_coloss
        return bct_coloss, bsc_coloss
