import numpy as np
import itertools
from .math import intersection_over_union

def generate_anchor_maps(C, image):

    image_shape = image.shape
    feature_pixels = C.feature_pixels

    assert len(image_shape) == 3, "Image must be 2D and RGB"

    areas = np.array(C.anchors_areas)
    aspect_ratios = np.array(C.aspect_ratios)
    anchor_sizes = np.array([ np.array([np.sqrt(i * j), np.sqrt(i / j)])  for(i, j) in itertools.product(areas, aspect_ratios) ])
    num_anchors = anchor_sizes.shape[0]

    anchor_template = np.zeros((anchor_sizes.shape[0], 4))
    anchor_template[:, 2:] = anchor_sizes / 2
    anchor_template[:, :2] = -anchor_sizes / 2

    image_height, image_width = image_shape[0], image_shape[1]
    height, width = image_shape[0] // feature_pixels, image_shape[1] // feature_pixels

    y_cell_coords = np.arange(height)
    x_cell_coords = np.arange(width)
    cell_coords = np.array(np.meshgrid(y_cell_coords, x_cell_coords)).T.reshape(-1, 2)

    center_points = (cell_coords * feature_pixels) + (feature_pixels // 2) # (H, W, 2)
    center_points = np.tile(center_points, reps = 2) # (H, W, 4)
    center_points = np.tile(center_points, reps = num_anchors) # (H, W, 4 * num_anchors)

    anchors = center_points.astype(np.float32) + anchor_template.flatten()
    anchors = anchors.reshape(height * width * num_anchors, 4) # (H * W * num_anchors, 4)

    valid = np.all((anchors[:, 0:2] >= [0, 0]) & (anchors[:, 2:4] < [image_height, image_width]), axis = 1)

    anchors = anchors.reshape((height, width, num_anchors * 4))
    valid = valid.reshape((height, width, num_anchors))

    return anchors.astype(np.float32), valid.astype(np.float32)

def generate_rpn_map(anchor_map, anchor_valid_map, gt_boxes, object_iou_threshold = 0.6, background_iou_threshold = 0.4):

    height, width, num_anchors = anchor_valid_map.shape

    gt_box_corners = np.array([ box["corners"] for box in gt_boxes ])
    num_gt_boxes = len(gt_boxes)

    # Compute GT box center points and side lengths
    gt_box_centers = 0.5 * (gt_box_corners[:, 0:2] + gt_box_corners[:, 2:4])
    gt_box_sides = gt_box_corners[:, 2:4] - gt_box_corners[:, 0:2]

    # Flatten anchor boxes to (N, 4) and convert to corners
    anchor_map = anchor_map.reshape((-1, 4))

    anchors = np.empty(anchor_map.shape)
    anchors[:, 0:2] = anchor_map[:, 0:2] - (0.5 * anchor_map[:, 2:4]) # y1, x1
    anchors[:, 2:4] = anchor_map[:, 0:2] + (0.5 * anchor_map[:, 2:4]) # y2, x2
    n = anchors.shape[0]

    # Initialize all anchors initially as negative
    objectness_score = np.full(n, -1) # RPN class: 0 -> background, 1 -> object, -1 -> ignore
    gt_box_assignments = np.full(n, -1) # -1 means no box

    # Compute IoU between all anchors and GT boxes
    ious = intersection_over_union(boxes1 = anchors, boxes2 = gt_box_corners)

    # Need to remove anchors that are invalid (outside image boundaries)
    ious[anchor_valid_map.flatten() == 0, :] = -1.0

    # Find the best IoU GT for each anchor and the best IoU anchor for each GT
    max_iou_per_anchor = np.max(ious, axis = 1)

    best_box_idx_per_anchor = np.argmax(ious, axis = 1)

    max_iou_per_gt_box = np.max(ious, axis = 0)
    highest_iou_anchor_idxs = np.where(ious == max_iou_per_gt_box)[0]

    # Anchors below the minimum threshold are negative
    objectness_score[max_iou_per_anchor < background_iou_threshold] = 0

    # Anchors that meet the threshold IoU are positive
    objectness_score[max_iou_per_anchor >= object_iou_threshold] = 1

    # Anchors that overlap the most with GT boxes are positive
    objectness_score[highest_iou_anchor_idxs] = 1

    # We assing the highest IoU GT box to each anchor. If no box met the IoU threshold, the 
    # highest IoU box may happen to be a box for which the anchor had the highest IoU. If not,
    # then tge objectness score will be negative and the box regression won't ever be used
    gt_box_assignments[:] = best_box_idx_per_anchor

    # Anchors that are to be ignored will be marked invalid. Generate a mask to multiply anchor_valid_map
    # by (-1 -> 0, 0 or 1 -> 1). Then mark ignored anchors as 0 in objectness score because the score
    # can only really be 0 or 1
    enable_mask = (objectness_score >= 0).astype(np.float32)
    objectness_score[objectness_score < 0] = 0

    # Compute box delta regression targets for each anchor
    box_delta_targets = np.zeros((n, 4))

    # ty = (box_center_y - anchor_center_y) / anchor_height, tx = (box_center_x - anchor_center_x) / anchor_width
    box_delta_targets[:, 0:2] = (gt_box_centers[gt_box_assignments] - anchor_map[:, 0:2]) / anchor_map[:, 2:4] 

    # th = log(box_height / anchor_height), tw = log(box_width / anchor_width)
    box_delta_targets[:, 2:4] = np.log(gt_box_sides[gt_box_assignments] / anchor_map[:, 2:4] + 1e-7)

    # Assemble RPN GT map
    rpn_map = np.zeros((height, width, num_anchors, 6))
    rpn_map[:, :, :, 0] = anchor_valid_map * enable_mask.reshape((height, width, num_anchors))
    rpn_map[:, :, :, 1] = objectness_score.reshape((height, width, num_anchors))
    rpn_map[:, :, :, 2:6] = box_delta_targets.reshape((height, width, num_anchors, 4))

    # Return map along with positive and negative anchors

    # Shape (height, width, k, 3): every index (y, x, k, :) returns its own coordinate (y, x, k)
    rpn_map_coords = np.transpose(np.mgrid[0:height, 0:width, 0:num_anchors], (1, 2, 3, 0))

    # Shape (N, 3) where each row is the coordinate (y, x, k) of a positive sample
    object_anchor_idxs = rpn_map_coords[np.where((rpn_map[:, :, :, 1] > 0) & (rpn_map[:, :, :, 0] > 0))]

    # Shape (N,3) where each row is the coordinate (y, x, k) of a negative sample
    background_anchor_idxs = rpn_map_coords[np.where((rpn_map[:, :, :, 1] == 0) & (rpn_map[:, :, :, 0] > 0))]

    return rpn_map.astype(np.float32), object_anchor_idxs, background_anchor_idxs