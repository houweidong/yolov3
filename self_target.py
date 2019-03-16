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


# class SelfDefaultTrainTransform(YOLO3DefaultTrainTransform):
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

    def __init__(self, width, height, net=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), mixup=False, **kwargs):
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
        self._target_generator = SelfPrefetchTargetGenerator(
            num_class=len(net.classes), **kwargs)

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
        objectness, center_targets, scale_targets, weights, class_targets, mask_obj, mask_cls = self._target_generator(
            self._fake_x, self._feat_maps, self._anchors, self._offsets,
            gt_bboxes, gt_ids, gt_mixratio)
        return (img, objectness[0], center_targets[0], scale_targets[0], weights[0],
                class_targets[0], mask_obj[0], mask_cls[0], gt_bboxes[0])


class SelfPrefetchTargetGenerator(gluon.Block):
    """Self prefetch target generator.
    The target generated by this instance is invariant to network predictions.
    Therefore it is usually used in DataLoader transform function to reduce the load on GPUs.

    Parameters
    ----------
    num_class : int
        Number of foreground classes.

    """
    def __init__(self, num_class, **kwargs):
        super(SelfPrefetchTargetGenerator, self).__init__(**kwargs)
        self._num_class = num_class
        self.bbox2center = BBoxCornerToCenter(axis=-1, split=True)
        self.bbox2corner = BBoxCenterToCorner(axis=-1, split=False)

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
        offsets = [nd.min_axis(offset, axis=-2, keepdims=False) for offset in offsets]
        all_anchors = nd.concat(*[a.reshape(-1, 2) for a in anchors], dim=0)
        assert isinstance(offsets, (list, tuple))
        all_offsets = nd.concat(*[o.reshape(-1, 2) for o in offsets], dim=0)
        num_anchors = np.cumsum([a.size // 2 for a in anchors])
        num_offsets = np.cumsum([o.size // 2 for o in offsets])
        _offsets = [0] + num_offsets.tolist()
        assert isinstance(xs, (list, tuple))
        assert len(xs) == len(anchors) == len(offsets)

        # orig image size
        orig_height = img.shape[2]
        orig_width = img.shape[3]
        with autograd.pause():
            # outputs
            # shape_like = all_anchors.reshape((1, -1, 2)) * all_offsets.reshape(
            #     (-1, 1, 2)).expand_dims(0).repeat(repeats=gt_ids.shape[0], axis=0)
            # center_targets = nd.zeros_like(shape_like)
            # scale_targets = nd.zeros_like(center_targets)
            # weights = nd.zeros_like(center_targets)
            # objectness = nd.zeros_like(weights.split(axis=-1, num_outputs=2)[0])
            # class_targets = nd.one_hot(objectness.squeeze(axis=-1), depth=self._num_class)
            # class_targets[:] = -1  # prefill -1 for ignores

            # output

            shape_like = all_anchors.reshape((1, -1, 2)) * all_offsets.reshape(
                (-1, 1, 2)).expand_dims(0).repeat(repeats=gt_ids.shape[0], axis=0)
            center_targets = nd.zeros_like(shape_like)
            scale_targets = nd.zeros_like(center_targets)
            weights = nd.zeros_like(center_targets)
            objectness = nd.zeros_like(weights.split(axis=-1, num_outputs=2)[0])
            mask_obj = nd.ones_like(objectness) * 1000
            distance = nd.ones_like(objectness) * 1000
            # class_targets = nd.one_hot(objectness.squeeze(axis=-1), depth=self._num_class)
            # class_targets[:] = 1000  # prefill 1000 for ignores
            mask_cls = nd.one_hot(objectness.squeeze(axis=-1), depth=self._num_class)
            mask_cls[:] = 1000  # prefill 1000 for ignores
            # distance_class = nd.ones_like(mask_cls) * 1000
            objectness_cls = nd.ones_like(mask_cls)

            # for each ground-truth, find the best matching anchor within the particular grid
            # for instance, center of object 1 reside in grid (3, 4) in (16, 16) feature map
            # then only the anchor in (3, 4) is going to be matched
            gtx, gty, gtw, gth = self.bbox2center(gt_boxes)
            shift_gt_boxes = nd.concat(-0.5 * gtw, -0.5 * gth, 0.5 * gtw, 0.5 * gth, dim=-1)
            anchor_boxes = nd.concat(0 * all_anchors, all_anchors, dim=-1)  # zero center anchors
            shift_anchor_boxes = self.bbox2corner(anchor_boxes)
            ious = nd.contrib.box_iou(shift_anchor_boxes, shift_gt_boxes).transpose((1, 0, 2))
            # real value is required to process, convert to Numpy
            matches = ious.argmax(axis=1).asnumpy()  # (B, M)
            valid_gts = (gt_boxes >= 0).asnumpy().prod(axis=-1)  # (B, M)
            np_gtx, np_gty, np_gtw, np_gth = [x.asnumpy() for x in [gtx, gty, gtw, gth]]
            np_anchors = all_anchors.asnumpy()
            np_gt_ids = gt_ids.asnumpy()
            np_gt_mixratios = gt_mixratio.asnumpy() if gt_mixratio is not None else None
            # TODO(zhreshold): the number of valid gt is not a big number, therefore for loop
            # should not be a problem right now. Switch to better solution is needed.
            for b in range(matches.shape[0]):
                for m in range(matches.shape[1]):
                    if valid_gts[b, m] < 1:
                        break
                    match = int(matches[b, m])
                    nlayer = np.nonzero(num_anchors > match)[0][0]
                    height = xs[nlayer].shape[2]
                    width = xs[nlayer].shape[3]

                    gtx, gty, gtw, gth = (np_gtx[b, m, 0], np_gty[b, m, 0],
                                          np_gtw[b, m, 0], np_gth[b, m, 0])
                    # compute the location of the gt centers
                    loc_x_point = gtx / orig_width * width
                    loc_y_point = gty / orig_height * height
                    # loc_x, loc_y = int(loc_x_point), int(loc_y_point)
                    # write back to targets

                    # make seal for label level
                    grid_x, grid_y = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
                    grid_x, grid_y = nd.array(grid_x).reshape(-1), nd.array(grid_y).reshape(-1)
                    distance_x, distance_y = nd.abs(grid_x - loc_x_point), nd.abs(grid_y - loc_y_point)
                    distance_max = nd.array(nd.maximum(distance_x, distance_y))
                    seal = nd.clip(nd.ceil(distance_max * 2), 1, 1000)  # 1000 just represent inf

            #         index = _offsets[nlayer] + loc_y * width + loc_x
            #         center_targets[b, index, match, 0] = gtx / orig_width * width - loc_x  # tx
            #         center_targets[b, index, match, 1] = gty / orig_height * height - loc_y  # ty
            #         scale_targets[b, index, match, 0] = np.log(gtw / np_anchors[match, 0])
            #         scale_targets[b, index, match, 1] = np.log(gth / np_anchors[match, 1])
            #         weights[b, index, match, :] = 2.0 - gtw * gth / orig_width / orig_height
            #         objectness[b, index, match, 0] = (
            #             np_gt_mixratios[b, m, 0] if np_gt_mixratios is not None else 1)
            #         class_targets[b, index, match, :] = 0
            #         class_targets[b, index, match, int(np_gt_ids[b, m, 0])] = 1
            # # since some stages won't see partial anchors, so we have to slice the correct targets
            # objectness = self._slice(objectness, num_anchors, num_offsets)
            # center_targets = self._slice(center_targets, num_anchors, num_offsets)
            # scale_targets = self._slice(scale_targets, num_anchors, num_offsets)
            # weights = self._slice(weights, num_anchors, num_offsets)
            # class_targets = self._slice(class_targets, num_anchors, num_offsets)
                    index = slice(_offsets[nlayer], _offsets[nlayer + 1])
                    cond = distance_max < distance[b, index, match, 0]
                    tx = loc_x_point - grid_x
                    ty = loc_y_point - grid_y
                    tw = nd.ones_like(cond) * np.log(max(gtw, 1) / np_anchors[match, 0])
                    th = nd.ones_like(cond) * np.log(max(gth, 1) / np_anchors[match, 1])
                    twg = nd.ones_like(cond) * (2.0 - gtw * gth / orig_width / orig_height)
                    tobj = nd.ones_like(cond) * (np_gt_mixratios[b, m, 0] if np_gt_mixratios is not None else 1)

                    center_targets[b, index, match, 0] = nd.where(cond, tx, center_targets[b, index, match, 0])
                    center_targets[b, index, match, 1] = nd.where(cond, ty, center_targets[b, index, match, 1])
                    scale_targets[b, index, match, 0] = nd.where(cond, tw, scale_targets[b, index, match, 0])
                    scale_targets[b, index, match, 1] = nd.where(cond, th, scale_targets[b, index, match, 1])
                    weights[b, index, match, :] = nd.where(cond.expand_dims(-1),
                                                           twg.expand_dims(-1), weights[b, index, match, 0:1])
                    objectness[b, index, match, 0] = nd.where(cond, tobj, objectness[b, index, match, 0])
                    mask_obj[b, index, match, 0] = nd.where(cond, seal, mask_obj[b, index, match, 0])

                    # cond_class = distance_max < distance_class[b, index, match, int(np_gt_ids[b, m, 0])]
                    cond_class = cond.expand_dims(-1).repeat(repeats=self._num_class, axis=-1)
                    objectness_cls[b, index, match, :] = \
                        nd.where(cond_class, nd.ones_like(cond_class), objectness_cls[b, index, match, :])
                    objectness_cls[b, index, match, int(np_gt_ids[b, m, 0])] = \
                        nd.where(cond, tobj, objectness_cls[b, index, match, int(np_gt_ids[b, m, 0])])
                    mask_cls[b, index, match, :] = \
                        nd.where(cond_class, nd.ones_like(cond_class) * 1000, mask_cls[b, index, match, :])
                    mask_cls[b, index, match, int(np_gt_ids[b, m, 0])] = \
                        nd.where(cond, seal, mask_cls[b, index, match, int(np_gt_ids[b, m, 0])])
                    distance[b, index, match, 0] = nd.where(cond, distance_max, distance[b, index, match, 0])
                    # class_targets[b, index, match, int(np_gt_ids[b, m, 0])] = \
                    #     nd.where(cond_class, seal, class_targets[b, index, match, int(np_gt_ids[b, m, 0])])
                    # distance_class[b, index, match, int(np_gt_ids[b, m, 0])] = \
                    #     nd.where(cond_class, distance_max, distance_class[b, index, match, int(np_gt_ids[b, m, 0])])
                # since some stages won't see partial anchors, so we have to slice the correct targets
                objectness = self._slice(objectness, num_anchors, num_offsets)
                center_targets = self._slice(center_targets, num_anchors, num_offsets)
                scale_targets = self._slice(scale_targets, num_anchors, num_offsets)
                weights = self._slice(weights, num_anchors, num_offsets)
                # class_targets = self._slice(class_targets, num_anchors, num_offsets)
                mask_obj = self._slice(mask_obj, num_anchors, num_offsets)
                objectness_cls = self._slice(objectness_cls, num_anchors, num_offsets)
                mask_cls = self._slice(mask_cls, num_anchors, num_offsets)
        return objectness, center_targets, scale_targets, weights, mask_cls, mask_obj, objectness_cls

    def _slice(self, x, num_anchors, num_offsets):
        """since some stages won't see partial anchors, so we have to slice the correct targets"""
        # x with shape (B, N, A, 1 or 2)
        anchors = [0] + num_anchors.tolist()
        offsets = [0] + num_offsets.tolist()
        ret = []
        for i in range(len(num_anchors)):
            y = x[:, offsets[i]:offsets[i+1], anchors[i]:anchors[i+1], :]
            ret.append(y.reshape((0, -3, -1)))
        return nd.concat(*ret, dim=1)
    

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
    def __init__(self, num_class, ignore_iou_thresh, **kwargs):
        super(SelfDynamicTargetGeneratorSimple, self).__init__(**kwargs)
        self._num_class = num_class
        self._ignore_iou_thresh = ignore_iou_thresh
        self._batch_iou = BBoxBatchIOU()

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
            objectness: 0 for negative, 1 for positive, -1 for ignore.
            center_targets: regression target for center x and y.
            scale_targets: regression target for scale x and y.
            weights: element-wise gradient weights for center_targets and scale_targets.
            class_targets: a one-hot vector for classification.

        """
        with autograd.pause():
            batch_ious = self._batch_iou(box_preds, gt_boxes)  # (B, N, M)
            ious_max = batch_ious.max(axis=-1, keepdims=True)  # (B, N, 1)
            objness_t = (ious_max > self._ignore_iou_thresh) * -1  # use -1 for ignored
        return objness_t
        # , center_t, scale_t, weight_t, class_t
