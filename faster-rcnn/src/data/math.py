import numpy as np
import tensorflow as tf

# Given two np.ndarrays of shape (N, 4) and (M, 4), returns an np.ndarray of shape (N, M) where the (i, j)-th entry is the IoU of the i-th box in boxes1 and the j-th box in boxes2.
def intersection_over_union(boxes1, boxes2):

    top_left_point = np.maximum(boxes1[:, None, 0:2], boxes2[:, 0:2])
    bottom_right_point = np.minimum(boxes1[:, None, 2:4], boxes2[:, 2:4])
    well_ordered_mask = np.all(top_left_point < bottom_right_point, axis = 2)

    intersection_areas = well_ordered_mask * np.prod(bottom_right_point - top_left_point, axis = 2)
    areas1 = np.prod(boxes1[:, 2:4] - boxes1[:, 0:2], axis = 1)
    areas2 = np.prod(boxes2[:, 2:4] - boxes2[:, 0:2], axis = 1)

    union_areas = areas1[:, None] + areas2 - intersection_areas

    return intersection_areas / (union_areas + 1e-7)

# IoU with TensorFlow
def tf_intersection_over_union(boxes1, boxes2):
  
  # Tile boxes2 and repeat boxes1. This allows us to compare every boxes1 against every boxes2 without loops.
  b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
  b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])

  # Compute intersections
  b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis = 1)
  b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis = 1)
  y1 = tf.maximum(b1_y1, b2_y1)
  x1 = tf.maximum(b1_x1, b2_x1)
  y2 = tf.minimum(b1_y2, b2_y2)
  x2 = tf.minimum(b1_x2, b2_x2)
  intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
  
  # Compute unions
  b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
  b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
  union = b1_area + b2_area - intersection

  # Compute IoU and reshape to [boxes1, boxes2]
  iou = intersection / union
  overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])

  return overlaps

# Converts box deltas (ty, tx, th, tw) to boxes (y1, x1, y2, x2).
def convert_deltas_to_boxes(box_deltas, anchors, box_delta_means, box_delta_stds):
    
    box_deltas = (box_deltas * box_delta_stds) + box_delta_means
    center = (anchors[:, 2:4] * box_deltas[:, 0:2]) + box_deltas[:, 0:2]
    size = anchors[:, 2:4] * np.exp(box_deltas[:, 2:4])

    boxes = np.empty(box_deltas.shape)
    boxes[:, 0:2] = center - (size / 2)
    boxes[:, 2:4] = center + (size / 2)

    return boxes

# Convert box deltas to boxes with TensorFlow
def tf_convert_deltas_to_boxes(box_deltas, anchors, box_delta_means, box_delta_stds):
  
  box_deltas = box_deltas * box_delta_stds + box_delta_means
  center = anchors[:, 2:4] * box_deltas[:, 0:2] + anchors[:, 0:2]
  size = anchors[:, 2:4] * tf.math.exp(box_deltas[:, 2:4])

  boxes_top_left = center - 0.5 * size
  boxes_bottom_right = center + 0.5 * size
  boxes = tf.concat([ boxes_top_left, boxes_bottom_right ], axis = 1)

  return boxes
