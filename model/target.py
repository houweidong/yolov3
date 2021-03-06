# from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
import numpy as np
import copy
from mxnet import gluon
from mxnet import nd
from mxnet import autograd
from gluoncv.nn.bbox import BBoxCornerToCenter, BBoxCenterToCorner, BBoxBatchIOU
import mxnet as mx
from gluoncv.data.transforms import bbox as tbbox
from gluoncv.data.transforms import image as timage
from gluoncv.data.transforms import experimental

class SelfDefaultTrainTransform(object):
    """Default YOLO training transform which includes tons of image augmentations.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    net : mxnet.gluon.HybridBlock, optional
        The yolo network.

        .. hint::

            If net is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.

    """

    def __init__(self, width, height, net=None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), mixup=False,
                 coop_configs=None, margin=0.5, thre_cls=None, coop_loss=False, coop_mode='flat', sigma_weight=None,
                 separate=False, label_smooth=True, **kwargs):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std
        self._mixup = mixup
        self._target_generator = None
        if net is None:
            return

        # in case network has reset_ctx to gpu
        self._fake_x = mx.nd.zeros((1, 3, height, width))
        net = copy.deepcopy(net)
        net.collect_params().reset_ctx(None)
        with autograd.train_mode():
            self._anchors, self._offsets, self._feat_maps = net(self._fake_x)
        self._target_generator = SelfPrefetchTargetGenerator(num_class=len(net.classes), coop_configs=coop_configs,
            margin=margin,thre_cls=thre_cls, coop_loss=coop_loss, coop_mode=coop_mode, sigma_weight=sigma_weight,
            separate=separate, label_smooth=label_smooth, **kwargs)

    def set_prob_fit(self, prob_fit=False):
        """Set mixup random sampler, use None to disable.

        Parameters
        ----------
        prob_fit : to make probability fit with the predicted boxes

        """
        self._target_generator.set_prob_fit(prob_fit=prob_fit)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # random color jittering
        img = experimental.image.random_color_distort(src)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
            bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = img, label

        # random cropping
        h, w, _ = img.shape
        bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = timage.imresize(img, self._width, self._height, interp=interp)
        bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._target_generator is None:
            return img, bbox.astype(img.dtype)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        if self._mixup:
            gt_mixratio = mx.nd.array(bbox[np.newaxis, :, -1:])
        else:
            gt_mixratio = None
        # center_targets, scale_targets, box_index, objectness, weights, mask_cls, weights_bl, mask_vtc, mask_hrz
        return_list = self._target_generator(self._fake_x, self._feat_maps, self._anchors, self._offsets, gt_bboxes, gt_ids, gt_mixratio)
        # return (img, center_targets[0], scale_targets[0], box_index[0], objectness[0],
        #         weights[0], mask_cls[0], weights_bl[0],  mask_vtc[0], mask_hrz[0], gt_bboxes[0])
        return [img] + [rt[0] for rt in return_list] + [gt_bboxes[0]]


class SelfPrefetchTargetGenerator(gluon.Block):
    """Self prefetch target generator.
    The target generated by this instance is invariant to network predictions.
    Therefore it is usually used in DataLoader transform function to reduce the load on GPUs.

    Parameters
    ----------
    num_class : int
        Number of foreground classes.
    prob_fit : bool
        Whether git the probability of box with iou between gt and the predicted boxes.
    coop_mode : string
        "flat", different level grids have same weight loss in the training phase
        "convex", the center grids have higher weight than the marginal grids in the training phase
        "concave", the marginal grids have higher weight than the center grids in the training phase

    """

    def __init__(self, num_class, prob_fit=False, coop_configs=None, margin=None, thre_cls=None,
                 coop_loss=False, coop_mode='flat', sigma_weight=None, separate=False, label_smooth=True, **kwargs):
        super(SelfPrefetchTargetGenerator, self).__init__(**kwargs)
        self._prob_fit = prob_fit
        # self._equal_train = equal_train
        self._num_class = num_class
        self.bbox2center = BBoxCornerToCenter(axis=-1, split=True)
        self.bbox2corner = BBoxCenterToCorner(axis=-1, split=False)
        self._coop_configs = coop_configs[::-1]
        self._max_thickness = coop_configs.shape[-1]
        self._threshold_cls = thre_cls
        # if less than 3, the len is 1
        self._margin = margin
        self._coop_loss = coop_loss
        self._coop_mode = coop_mode
        self._sigma_weight = sigma_weight
        self._separate = separate
        self._label_smooth = label_smooth

    def set_prob_fit(self, prob_fit=False):
        """Set mixup random sampler, use None to disable.

        Parameters
        ----------
        prob_fit : to make probability fit with the predicted boxes

        """
        self._prob_fit = prob_fit

    def forward(self, img, xs, anchors, offsets, gt_boxes, gt_ids, gt_mixratio=None):
        """Generating training targets that do not require network predictions.

        Parameters
        ----------
        img : mxnet.nd.NDArray
            Original image tensor.
        xs : list of mxnet.nd.NDArray
            List of feature maps.
        anchors : mxnet.nd.NDArray
            YOLO3 anchors.
        offsets : mxnet.nd.NDArray
            Pre-generated x and y offsets for YOLO3.
        gt_boxes : mxnet.nd.NDArray
            Ground-truth boxes.
        gt_ids : mxnet.nd.NDArray
            Ground-truth IDs.
        gt_mixratio : mxnet.nd.NDArray, optional
            Mixup ratio from 0 to 1.

        Returns
        -------
        (tuple of) mxnet.nd.NDArray
            objectness: 0 for negative, 1 for positive, -1 for ignore.
            center_targets: regression target for center x and y.
            scale_targets: regression target for scale x and y.
            weights: element-wise gradient weights for center_targets and scale_targets.
            class_targets: a one-hot vector for classification.

        """
        assert isinstance(anchors, (list, tuple))
        # offsets = [nd.min_axis(offset, axis=-2, keepdims=False) for offset in offsets]
        all_anchors = nd.concat(*[a.reshape(-1, 2) for a in anchors], dim=0)
        assert isinstance(offsets, (list, tuple))
        all_offsets = nd.concat(*[o.reshape(-1, ) for o in offsets], dim=0)
        num_anchors = np.cumsum([a.size // 2 for a in anchors])
        num_offsets = np.cumsum([o.size for o in offsets])
        _offsets = [0] + num_offsets.tolist()
        assert isinstance(xs, (list, tuple))
        assert len(xs) == len(anchors) == len(offsets)

        # orig image size
        orig_height = img.shape[2]
        orig_width = img.shape[3]
        with autograd.pause():

            # output
            # shape_like = all_anchors.reshape((1, -1, 2)) * all_offsets.reshape(
            #     (-1, 1, 2)).expand_dims(0).repeat(repeats=gt_ids.shape[0], axis=0)
            #
            # center_targets = nd.zeros_like(shape_like)
            center_targets = nd.zeros(shape=(gt_ids.shape[0], all_offsets.shape[0], 3, 2))
            scale_targets = nd.zeros_like(center_targets)
            weights = nd.zeros_like(center_targets)
            objectness = nd.zeros_like(weights.split(axis=-1, num_outputs=2)[0])
            # mask_obj = nd.ones_like(objectness) * 1000
            distance = nd.ones_like(objectness) * 1000
            # mask_cls = nd.one_hot(objectness.squeeze(axis=-1), depth=self._num_class)
            # mask_cls[:] = 1000  # prefill 1000 for ignores
            mask_cls = nd.zeros_like(objectness)
            box_index = nd.ones_like(objectness) * -1
            weights_bl = nd.ones_like(objectness.expand_dims(-2).repeat(repeats=self._max_thickness, axis=-2)) * 1000
            distance_margin = nd.ones_like(objectness) * 1000
            if self._coop_loss:
                mask_vtc, mask_hrz = nd.zeros_like(weights_bl), nd.zeros_like(weights_bl)
            # for each ground-truth, find the best matching anchor within the particular grid
            # for instance, center of object 1 reside in grid (3, 4) in (16, 16) feature map
            # then only the anchor in (3, 4) is going to be matched
            gtx, gty, gtw, gth = self.bbox2center(gt_boxes)
            shift_gt_boxes = nd.concat(-0.5 * gtw, -0.5 * gth, 0.5 * gtw, 0.5 * gth, dim=-1)
            anchor_boxes = nd.concat(0 * all_anchors, all_anchors, dim=-1)  # zero center anchors
            shift_anchor_boxes = self.bbox2corner(anchor_boxes)
            ious = nd.contrib.box_iou(shift_anchor_boxes, shift_gt_boxes).transpose((1, 0, 2))
            # real value is required to process, convert to Numpy
            # print(nd.softmax(ious, axis=1))
            # print(ious[0] / nd.sum(ious[0], axis=0, keepdims=True))
            # print(nd.softmax(ious[0], axis=0))
            # print(ious[0])
            matches = ious.argmax(axis=1).asnumpy()  # (B, M)
            valid_gts = (gt_boxes >= 0).asnumpy().prod(axis=-1)  # (B, M)
            np_gtx, np_gty, np_gtw, np_gth = [x.asnumpy() for x in [gtx, gty, gtw, gth]]
            np_anchors = all_anchors.asnumpy()
            np_gt_ids = gt_ids.asnumpy()
            np_gt_mixratios = gt_mixratio.asnumpy() if gt_mixratio is not None else None
            # TODO(zhreshold): the number of valid gt is not a big number, therefore for loop
            # should not be a problem right now. Switch to better solution is needed.
            # t_or_f_h, t_or_f_v = False, False
            for b in range(matches.shape[0]):
                for m in range(matches.shape[1]):
                    if valid_gts[b, m] < 1:
                        break
                    match9 = int(matches[b, m])
                    match = match9 % 3
                    nlayer = np.nonzero(num_anchors > match9)[0][0]
                    height = xs[nlayer].shape[2]
                    width = xs[nlayer].shape[3]

                    gtx, gty, gtw, gth = (np_gtx[b, m, 0], np_gty[b, m, 0],
                                          np_gtw[b, m, 0], np_gth[b, m, 0])
                    # compute the location of the gt centers
                    loc_x_point = gtx / orig_width * width
                    loc_y_point = gty / orig_height * height
                    # loc_x, loc_y = int(loc_x_point), int(loc_y_point)
                    # write back to targets

                    # if loc_x_point % 1 == 0:
                    #     t_or_f_h = True
                    # if loc_y_point % 1 == 0:
                    #     t_or_f_v = True

                    # make seal for label level
                    # print(loc_x_point, loc_y_point)
                    grid_x, grid_y = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
                    grid_x, grid_y = nd.array(grid_x).reshape(-1), nd.array(grid_y).reshape(-1)
                    distance_x, distance_y = nd.abs(grid_x - loc_x_point), nd.abs(grid_y - loc_y_point)
                    distance_max = nd.array(nd.maximum(distance_x, distance_y))
                    # seal = nd.clip(nd.ceil(distance_max * 2), 1, 1000)  # 1000 just represent inf
                    dis = nd.sqrt(nd.square(distance_x) + nd.square(distance_y))

                    index = slice(_offsets[nlayer], _offsets[nlayer + 1])
                    cond = dis < distance[b, index, match, 0]

                    dis_margin = nd.abs(distance[b, index, match, 0] - dis)
                    cond_mg = ((dis_margin < distance_margin[b, index, match, 0]) + cond)
                    distance_margin[b, index, match, 0] = nd.where(cond_mg, dis_margin, distance_margin[b, index, match, 0])
                    tx = loc_x_point - grid_x
                    ty = loc_y_point - grid_y
                    tw = nd.ones_like(cond) * np.log(max(gtw, 1) / np_anchors[match9, 0])
                    th = nd.ones_like(cond) * np.log(max(gth, 1) / np_anchors[match9, 1])
                    twg = nd.ones_like(cond) * (2.0 - gtw * gth / orig_width / orig_height)
                    tobj = nd.ones_like(cond) * (np_gt_mixratios[b, m, 0] if np_gt_mixratios is not None else 1)

                    center_targets[b, index, match, 0] = nd.where(cond, tx, center_targets[b, index, match, 0])
                    center_targets[b, index, match, 1] = nd.where(cond, ty, center_targets[b, index, match, 1])
                    scale_targets[b, index, match, 0] = nd.where(cond, tw, scale_targets[b, index, match, 0])
                    scale_targets[b, index, match, 1] = nd.where(cond, th, scale_targets[b, index, match, 1])
                    weights[b, index, match, :] = nd.where(cond.expand_dims(-1),
                                                           twg.expand_dims(-1), weights[b, index, match, 0:1])
                    objectness[b, index, match, 0] = nd.where(cond, tobj, objectness[b, index, match, 0])
                    # mask_obj[b, index, match, 0] = nd.where(cond, distance_max, mask_obj[b, index, match, 0])
                    weights_bl[b, index, match, :, 0] = nd.where(cond.expand_dims(-1), distance_max.expand_dims(-1),
                                                                 weights_bl[b, index, match, 0:1, 0])
                    # cond_class = cond.expand_dims(-1).tile(reps=(self._num_class, ))
                    # mask_cls[b, index, match, :] = \
                    #     nd.where(cond_class, nd.ones_like(cond_class) * 1000, mask_cls[b, index, match, :])
                    # mask_cls[b, index, match, int(np_gt_ids[b, m, 0])] = \
                    #     nd.where(cond, distance_max, mask_cls[b, index, match, int(np_gt_ids[b, m, 0])])
                    mask_cls[b, index, match, 0] = nd.where(cond, nd.ones_like(cond) * int(np_gt_ids[b, m, 0]), mask_cls[b, index, match, 0])

                    # if self._prob_fit:
                    box_index[b, index, match, 0] = nd.where(cond, nd.ones_like(cond) * m, box_index[b, index, match, 0])
                    # distance[b, index, match, 0] = nd.where(cond, distance_max, distance[b, index, match, 0])
                    distance[b, index, match, 0] = nd.where(cond, dis, distance[b, index, match, 0])
            # to modify the mask for margin the grid
            # for i, margins in enumerate(self._margin):
            #     index = slice(_offsets[i], _offsets[i + 1])
            #     for j, margin in enumerate(margins):
            #         cond_modify = distance_margin[:, index, j, 0] < margin
            #         mask_obj[:, index, j, 0] = nd.where(cond_modify, nd.ones_like(cond_modify) * 1000, mask_obj[:, index, j, 0])

            # 100 just for long enough
            # weight_default = np.arange(1, 100, 2)
            # box_index_np = box_index.asnumpy()
            # mask_obj_np = mask_obj.asnumpy()

            for i, cfgss in enumerate(self._coop_configs):
                index = slice(_offsets[i], _offsets[i + 1])
                for j, cfgs in enumerate(cfgss):
                    cond_modify_cls = (weights_bl[:, index, j, 0, :] > self._threshold_cls[i, j, 0] / 2)  # + \
                    # ((distance_margin[:, index, j, :] <= min(self._margin[i, j])) * (weights_bl[:, index, j, 0, :] > 0.5))
                    mask_cls[:, index, j, :] = mask_cls[:, index, j, :] - cond_modify_cls * 100
                    for k, cfg in enumerate(cfgs):
                        cond_modify = (weights_bl[:, index, j, k, :] <= cfg / 2)  # * ((distance_margin[:, index, j, :]
                        # > self._margin[i, j, k]) + (weights_bl[:, index, j, k, :] <= 0.5))
                        fct_margin = nd.where(distance_margin[:, index, j, :] > self._margin[i, j, k], nd.ones_like(
                            distance_margin[:, index, j, :]), distance_margin[:, index, j, :] / self._margin[i, j, k])
                        fct_margin = nd.power(fct_margin, 1.5)
                        if self._coop_mode == 'flat':
                            weights_bl[:, index, j, k, :] = nd.where(cond_modify,
                                nd.ones_like(cond_modify)/(cfg*cfg) * fct_margin, nd.zeros_like(cond_modify))
                        elif self._coop_mode == 'convex':
                            guassian_dis = nd.exp(-1 * distance[:, index, j, :] / (self._sigma_weight[i, j, k] ** 2)) * fct_margin / (cfg ** 0.8)
                            weights_bl[:, index, j, k, :] = nd.where(cond_modify, guassian_dis, nd.zeros_like(cond_modify))
                        else:
                            raise Exception('only support flat, convex now')

            # print(matches.shape[1])
            # from matplotlib import pyplot
            # for i in range(3):
            #     index = slice(_offsets[i], _offsets[i + 1])
            #     side = int(np.sqrt(_offsets[i + 1] - _offsets[i]))
            #     a = weights_bl.asnumpy()[0, index, :, 0, 0].reshape(side, side, 3)
            #     pyplot.matshow(a[:, :, 0])
            #     pyplot.matshow(a[:, :, 1])
            #     pyplot.matshow(a[:, :, 2])
            # pyplot.show()
            if self._coop_loss:
                masked_index = nd.where(weights_bl == 0, -nd.ones_like(weights_bl), box_index.expand_dims(-2).repeat(repeats=self._max_thickness, axis=-2))
                for i, cfgss in enumerate(self._coop_configs):
                    index = slice(_offsets[i], _offsets[i + 1])
                    side = int(np.sqrt(_offsets[i + 1] - _offsets[i]))
                    box_reshape = (masked_index[:, index, :, :, :]).reshape((0, -4, side, side, 0, -3))
                    box_vtc = nd.concat(box_reshape[:, 1:, :, :, :], box_reshape[:, 0:1, :, :, :], dim=1).reshape((0, -3, 0, 0, 1))
                    box_hrz = nd.concat(box_reshape[:, :, 1:, :, :], box_reshape[:, :, 0:1, :, :], dim=2).reshape((0, -3, 0, 0, 1))
                    mask_vtc[:, index, :, :, :] = (box_vtc == masked_index[:, index, :, :, :])
                    mask_hrz[:, index, :, :, :] = (box_hrz == masked_index[:, index, :, :, :])
            # if self._coop_loss:
            #     v = ((mask_vtc * weights_bl) != 0).sum().asscalar()
            #     h = ((mask_hrz * weights_bl) != 0).sum().asscalar()
                # print(t_or_f_v, t_or_f_h, v, h)

            # if self._equal_train:
            #     # TODO equal_train
            #     pass
                # for b in range(matches.shape[0]):
                #     for m in range(matches.shape[1]):
                #         if valid_gts[b, m] < 1:
                #             break
                #
                #         # use match to reduce time but box_index_np[b, index, :, 0]
                #         match9 = int(matches[b, m])
                #         match = match9 % 3
                #         nlayer = np.nonzero(num_anchors > match9)[0][0]
                #         index = slice(_offsets[nlayer], _offsets[nlayer + 1])
                #         max_level = self._coop_configs[nlayer, -1, -1]
                #         cond = (box_index_np[b, index, match, 0] == m) & (mask_obj_np[b, index, match, 0] <= max_level)
                #         his, _ = np.histogram(mask_obj_np[b, index, match, 0][cond], [i+1 for i in range(max_level)] + [max_level],
                #                               (1, max_level))
                #         for index_cfg, cfg in enumerate(self._coop_configs[nlayer]):
                #             # this target with the cfg has been covered by other target
                #             if sum(his[:cfg]) == 0:
                #                 continue
                #
                #             # old is bad
                #             #  = np.where(his[:cfg] == 0, 0, weight_default[:cfg] / np.where(his[:cfg] == 0, 1, his[:cfg]))
                #             # need_add = sum(weight_default[:cfg][his[:cfg] == 0])
                #             # weight_list = weight_list * (need_add / (cfg ** 2 - need_add) + 1)
                #             # if not self._equal_train:
                #             #     weight_list[:] = 1
                #             # weights_bl[b, index, match, index_cfg, 0] = np.select([cond & (mask_obj_np[b, index, match, 0]
                #             #     == i+1) for i in range(cfg)], weight_list, weights_bl[b, index, match, index_cfg, 0])
                #
                #             # new to try
                #             weight_list = np.ones_like(his[:cfg]) * (cfg**2) / sum(his[:cfg])
                #             if not self._equal_train:
                #                 weight_list[:] = 1
                #             weights_bl[b, index, match, index_cfg, 0] = np.select([cond & (mask_obj_np[b, index, match, 0]
                #                 == i+1) for i in range(cfg)], weight_list, weights_bl[b, index, match, index_cfg, 0])
            objectness = objectness.reshape((0, -3, 1, -1))
            center_targets = center_targets.reshape((0, -3, 1, -1))
            scale_targets = scale_targets.reshape((0, -3, 1, -1))
            weights = weights.reshape((0, -3, 1, -1))
            box_index = box_index.reshape((0, -3, 1, -1))
            weights_bl = weights_bl.reshape((0, -3, 0, 0))
            mask_cls = mask_cls.reshape((0, -3, 1, -1))
            obj = objectness * weights_bl
            weights = obj * weights
            if self._coop_loss:
                mask_vtc = mask_vtc.reshape((0, -3, 0, 0)) * weights
                mask_hrz = mask_hrz.reshape((0, -3, 0, 0)) * weights
            fac_cls = weights_bl[:, :, -1:, :]
            if self._separate:
                obj = nd.where(mask_cls >= 0, obj, nd.zeros_like(obj))
                fac_cls = 1
            fctr_cls = nd.where(mask_cls >= 0, objectness * fac_cls, nd.zeros_like(objectness))
            mask_cls = nd.one_hot(mask_cls.squeeze(axis=-1), depth=self._num_class)
            if not self._prob_fit:
                box_index[:] = -1
            if self._label_smooth:
                smooth_weight = 1. / self._num_class
                mask_cls = nd.where(mask_cls > 0.5, mask_cls - smooth_weight, nd.ones_like(mask_cls) * smooth_weight)

        if self._coop_loss:
            return center_targets, scale_targets, box_index, obj, mask_cls, fctr_cls, weights, mask_vtc, mask_hrz
        else:
            return center_targets, scale_targets, box_index, obj, mask_cls, fctr_cls, weights


class SelfDynamicTargetGeneratorSimple(gluon.HybridBlock):
    """YOLOV3 target generator that requires network predictions.
    `Dynamic` indicate that the targets generated depend on current network.
    `Simple` indicate that it only support `pos_iou_thresh` >= 1.0,
    otherwise it's a lot more complicated and slower.
    (box regression targets and class targets are not necessary when `pos_iou_thresh` >= 1.0)

    Parameters
    ----------
    num_class : int
        Number of foreground classes.
    ignore_iou_thresh : float
        Anchors that has IOU in `range(ignore_iou_thresh, pos_iou_thresh)` don't get
        penalized of objectness score.

    """

    def __init__(self, num_class, ignore_iou_thresh, len_coop, **kwargs):
        super(SelfDynamicTargetGeneratorSimple, self).__init__(**kwargs)
        self._num_class = num_class
        self._ignore_iou_thresh = ignore_iou_thresh
        self._batch_iou = BBoxBatchIOU()
        self._len_coop = len_coop

    def hybrid_forward(self, F, box_preds, gt_boxes):
        """Short summary.

        Parameters
        ----------
        F : mxnet.nd or mxnet.sym
            `F` is mxnet.sym if hybridized or mxnet.nd if not.
        box_preds : mxnet.nd.NDArray
            Predicted bounding boxes.
        gt_boxes : mxnet.nd.NDArray
            Ground-truth bounding boxes.

        Returns
        -------
        (tuple of) mxnet.nd.NDArray
            batch_ious: iou between current output layer with the gt boxes.
            objectness: 0 for negative, 1 for positive, -1 for ignore.

        """
        with autograd.pause():
            # (B, hwc, cfg, 1)
            batch_ious = self._batch_iou(box_preds, gt_boxes).reshape((0, -4, -1, self._len_coop, 0))
            ious_max = batch_ious.max(axis=-1, keepdims=True)
            objness_t = (ious_max > self._ignore_iou_thresh) * -1  # use -1 for ignored
        return F.stop_gradient(batch_ious), F.stop_gradient(objness_t)

# def get_factor(coop_configs, coop_mode, sigma_weight):
#     """
#     Parameters
#     ----------
#     coop_configs : tuple
#         current output layer's sigmoid configs, such as (1, 2, )
#     coop_mode : string
#         "flat", different level grids have same weight loss in the training phase
#         "convex", the center grids have higher weight than the marginal grids in the training phase
#         "concave", the marginal rids have higher weight than the center grids in the training phase
#         "equal", consider the num of the same level grids to make loss equal
#     sigma_weight : float
#         for coop_mode params, we use Gaussian distribution to generate the weights according to the grid level,
#         the sigma_weight is the distribution's params
#
#     """
#     factor_list, center_list = [], []
#     factor_max, config_max, center_max = None, 0, None
#     for config in coop_configs:
#         if coop_mode == 'flat':
#             factor_list.append(1 / (config ** 2))
#             if config > config_max:
#                 config_max = config
#                 factor_max = 1 / (config ** 2)
#         elif coop_mode in ['convex', 'concave']:
#             center = 1 if coop_mode == 'convex' else config
#             num_evry_grid = np.arange(1, 2*config, 2)
#             factors_gird = np.exp(-1 * np.square(np.arange(1, config + 1) - center) / (sigma_weight ** 2))
#             factor_same = 1 / ((factors_gird * num_evry_grid).sum())
#             factor_list.append(factor_same)
#             center_list.append(center)
#             if config > config_max:
#                 config_max = config
#                 factor_max = factor_same
#                 center_max = center
#         else:
#             factor_list.append(1 / config)
#             if config > config_max:
#                 config_max = config
#                 factor_max = 1 / config
#     return factor_list, center_list, factor_max, center_max
