import os
import cv2
import tensorflow as tf
import numpy as np
import random
import traceback
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from src.utils.csv_logs import CSVLog
from src.utils.weights_tracker import BestWeightsTracker
from src.utils.statistics import TrainingStatistics, PrecisionRecallCurveCalculator
from src.utils.visualize import show_detections
from src.models.faster_rcnn import FasterRCNNModel
from src.data.anchors import generate_anchor_maps
from src.data.image import load_image
from src.utils.config import Config
from src.data.dataset import Dataset

def _sample_rpn_minibatch(rpn_map, object_indices, background_indices, rpn_minibatch_size):

    assert rpn_map.shape[0] == 1, "Batch size must be 1"
    assert len(object_indices) == 1, "Batch size must be 1"
    assert len(background_indices) == 1, "Batch size must be 1"

    positive_anchors = object_indices[0]
    negative_anchors = background_indices[0]

    assert len(positive_anchors) + len(negative_anchors) >= rpn_minibatch_size, "Image has insufficient anchors for RPN minibatch size of %d" % rpn_minibatch_size
    assert len(positive_anchors) > 0, "Image does not have any positive anchors"
    assert rpn_minibatch_size % 2 == 0, "RPN minibatch size must be evenly divisible"

    # Sample, producing indices into the index maps
    num_positive_anchors = len(positive_anchors)
    num_negative_anchors = len(negative_anchors)
    num_positive_samples = min(rpn_minibatch_size // 2, num_positive_anchors) # up to half the samples should be positive, if possible
    num_negative_samples = rpn_minibatch_size - num_positive_samples # the rest should be negative
    positive_anchor_idxs = random.sample(range(num_positive_anchors), num_positive_samples)
    negative_anchor_idxs = random.sample(range(num_negative_anchors), num_negative_samples)

    # Construct index expressions into RPN map
    positive_anchors = positive_anchors[positive_anchor_idxs]
    negative_anchors = negative_anchors[negative_anchor_idxs]
    trainable_anchors = np.concatenate([ positive_anchors, negative_anchors ])
    batch_idxs = np.zeros(len(trainable_anchors), dtype = int)
    trainable_idxs = (batch_idxs, trainable_anchors[:, 0], trainable_anchors[:, 1], trainable_anchors[:, 2], 0)

    # Create a copy of the RPN map with samples set as trainable
    rpn_minibatch_map = rpn_map.copy()
    rpn_minibatch_map[:, :, :, :, 0] = 0
    rpn_minibatch_map[trainable_idxs] = 1

    return rpn_minibatch_map

def _convert_training_sample_to_model_input(sample, mode):

    # Ground truth boxes to NumPy arrays
    gt_box_corners = np.array([ box["corners"] for box in sample["gt_boxes"] ]).astype(np.float32) # (num_boxes, 4), where each row is (y1, x1, y2, x2)
    gt_box_class_idxs = np.array([ box["class_index"] for box in sample["gt_boxes"] ]).astype(np.int32) # (num_boxes, ), where each is an index (1, num_classes)

    # Expand all maps to a batch size of 1
    image_data = np.expand_dims(sample["image"], axis = 0)
    image_shape_map = np.array([ [ image_data.shape[1], image_data.shape[2], image_data.shape[3] ] ]) # (1, 3), with (height, width, channels)
    anchor_map = np.expand_dims(sample["anchor_map"], axis = 0)
    anchor_valid_map = np.expand_dims(sample["anchor_valid_map"], axis = 0)
    gt_rpn_map = np.expand_dims(sample["gt_rpn_map"], axis = 0)
    gt_rpn_object_indices = [ sample["gt_rpn_object_indices"] ]
    gt_rpn_background_indices = [ sample["gt_rpn_background_indices"] ]
    gt_box_corners = np.expand_dims(gt_box_corners, axis = 0)
    gt_box_class_idxs = np.expand_dims(gt_box_class_idxs, axis = 0)

    # Create a RPN minibatch: sample anchors randomly and create a new ground truth RPN map
    gt_rpn_minibatch_map = _sample_rpn_minibatch(
      rpn_map = gt_rpn_map,
      object_indices = gt_rpn_object_indices,
      background_indices = gt_rpn_background_indices,
      rpn_minibatch_size = 256
    )

    # Input vector to model
    if mode == "train": x = [ image_data, anchor_map, anchor_valid_map, gt_rpn_minibatch_map, gt_box_class_idxs, gt_box_corners ]
    # Infer
    else: x = [ image_data, anchor_map, anchor_valid_map ]

    # Return all plus some unpacked elements for convenience
    return x, image_data, gt_rpn_minibatch_map

def evaluate(model, eval_data = None, num_samples = None, plot = False, print_average_precisions = False):
    
    if num_samples is None: num_samples = eval_data.num_samples
    
    precision_recall_curve = PrecisionRecallCurveCalculator()
    i = 0

    print("Evaluating...")
    skipped = 0

    for sample in tqdm(iterable = iter(eval_data), total = num_samples):

        try:

            x, image_data, _ = _convert_training_sample_to_model_input(sample = sample, mode = "infer")
            scored_boxes_by_class_index = model.predict_on_batch(x = x, score_threshold = 0.05) # lower threshold score for evaluation
            precision_recall_curve.add_image_results(scored_boxes_by_class_index = scored_boxes_by_class_index, gt_boxes = sample["gt_boxes"])

        except:
            skipped += 1
            # traceback.print_exc()
            # print("Skipping image due to error.")
            continue

        i += 1
        if i >= num_samples: break

    print("Skipped %d images due to errors." % skipped)

    if print_average_precisions: precision_recall_curve.print_average_precisions(class_index_to_name = eval_data.class_index_to_name)

    mean_average_precision = 100.0 * precision_recall_curve.compute_mean_average_precision()
    print("Mean Average Precision = %1.2f%%" % mean_average_precision)
    if plot: precision_recall_curve.plot_average_precisions(class_index_to_name = eval_data.class_index_to_name)
    
    return mean_average_precision

def train(C, model, dataset_path, output_path, epochs, learning_rate, beta1, beta2, dropout):

    training_data = Dataset(C, dataset_path = dataset_path + "/train/", shuffle = True, cache = True)
    eval_data = Dataset(C, dataset_path = dataset_path + "/valid/", shuffle = False, cache = False)
    if not os.path.exists(output_path): os.makedirs(output_path)
    csv = CSVLog(output_path + "/log.csv")
    best_weights_tracker = BestWeightsTracker(filepath = output_path + "/best_weights.h5")

    for epoch in range(1, 1 + epochs):

        stats = TrainingStatistics()
        progbar = tqdm(iterable = iter(training_data), total = training_data.num_samples, postfix = stats.get_progbar_postfix())

        skipped = 0

        for sample in progbar:
            
            try:
                x, image_data, gt_rpn_minibatch_map = _convert_training_sample_to_model_input(sample = sample, mode = "train")
                losses = model.train_on_batch(x = x, y = gt_rpn_minibatch_map, return_dict = True)
                stats.on_training_step(losses = losses)
                progbar.set_postfix(stats.get_progbar_postfix())
            except:
                skipped += 1
                # print("Skipping image due to error.")
                continue

        print("Skipped %d images due to errors." % skipped)

        last_epoch = epoch == epochs
        mean_average_precision = evaluate(model = model, eval_data = eval_data, num_samples = None, plot = False, print_average_precisions = False)

        # Checkpoint
        checkpoint_file = os.path.join(output_path, "checkpoint-epoch-%d-mAP-%1.1f.h5" % (epoch, mean_average_precision))
        model.save_weights(filepath = checkpoint_file, overwrite = True, save_format = "h5")
        print("Saved model checkpoint to '%s'" % checkpoint_file)

        # CSV log
        log_items = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "dropout": dropout,
            "mAP": mean_average_precision
        }
        log_items.update(stats.get_progbar_postfix())
        csv.log(log_items)

        best_weights_tracker.on_epoch_end(model = model, mAP = mean_average_precision)

    model.save_weights(filepath = output_path + "/last_weights.h5", overwrite = True, save_format = "h5")
    print("Saved final model weights to '%s'" % (output_path + "/last_weights.h5"))

def _predict(model, image_data, image, show_image, output_path):
    
    anchor_map, anchor_valid_map = generate_anchor_maps(image_shape = image_data.shape, feature_pixels = 16)
    anchor_map = np.expand_dims(anchor_map, axis = 0) # convert to batch size of 1
    anchor_valid_map = np.expand_dims(anchor_valid_map, axis = 0)

    image_data = np.expand_dims(image_data, axis = 0) # convert to batch size of 1: (1, height, width, 3)
    image_shape_map = np.array([ [ image_data.shape[1], image_data.shape[2], image_data.shape[3] ] ]) # (1, 3), with (height, width, channels)

    x = [ image_data, anchor_map, anchor_valid_map ]

    scored_boxes_by_class_index = model.predict_on_batch(x = x, score_threshold = 0.7)
    show_detections(
        output_path = output_path,
        show_image = show_image,
        image = image,
        scored_boxes_by_class_index = scored_boxes_by_class_index,
        class_index_to_name = Dataset.class_index_to_name
    )

def predict_one(model, url, show_image, output_path):
  image_data, image, _, _ = load_image(url = url, max_dimension_pixels = 600)
  _predict(model = model, image_data = image_data, image = image, show_image = show_image, output_path = output_path)

C = Config()

# GPU settings
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Params
learning_rate = 0.0001
beta1 = 0.9
beta2 = 0.999

# Model
optimizer = Adam(learning_rate = learning_rate, beta_1 = beta1, beta_2 = beta2)
model = FasterRCNNModel(
    C,
    num_classes = 8,
    allow_edge_proposals = False,
    custom_roi_pool = False,
    activate_class_outputs = True,
    l2 = 0.5 * 0.0001,
    dropout_probability = 0.5,
)
model.build(
    input_shape = [
      (1, None, None, 3), # input_image: (1, height_pixels, width_pixels, 3)
      (1, None, None, 9 * 4), # anchor_map: (1, height, width, num_anchors * 4)
      (1, None, None, 9), # anchor_valid_map: (1, height, width, num_anchors)
      (1, None, None, 9, 6), # gt_rpn_map: (1, height, width, num_anchors, 6)
      (1, None), # gt_box_class_idxs_map: (1, num_gt_boxes)
      (1, None, 4) # gt_box_corners_map: (1, num_gt_boxes, 4)
    ]
  )
model.compile(optimizer = optimizer)

train(C, model, "./dataset", "./outputs", epochs = 10, learning_rate = learning_rate, beta1 = beta1, beta2 = beta2, dropout = 0.5)

# dataset = Dataset(C, "./dataset/train", 16)

# import matplotlib.pyplot as plt

# for data in tqdm(iterable = iter(dataset), total = dataset.num_samples):

#     print("Positives: ", len(data["gt_rpn_object_indices"]))
#     print("Negatives: ", len(data["gt_rpn_background_indices"]))
#     print("=====================================")

#     anchor_map = data["anchor_map"]
#     image = data["image"]
#     gt_boxes = data["gt_boxes"]

#     gt_box_image = image.copy()

#     for gt_box in gt_boxes:

#         corners = gt_box["corners"]
#         class_name = gt_box["class_name"]
#         class_index = gt_box["class_index"]
#         y1, x1, y2, x2 = corners.astype(int)

#         cv2.rectangle(gt_box_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         # cv2.putText(image, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     plt.imshow(gt_box_image)
#     plt.show()

#     gt_object_image = image.copy()

#     for (y, x, k) in data["gt_rpn_object_indices"]:

#         color = (0, 0, 255)

#         height = anchor_map[y,x,k*4+2]
#         width = anchor_map[y,x,k*4+3]
#         cy = anchor_map[y,x,k*4+0]
#         cx = anchor_map[y,x,k*4+1]

#         corners = np.array([cy - 0.5 * height, cx - 0.5 * width, cy + 0.5 * height, cx + 0.5 * width])
#         y1, x1, y2, x2 = corners.astype(int)

#         cv2.rectangle(gt_object_image, (x1, y1), (x2, y2), color, 1)

#     plt.imshow(gt_object_image)
#     plt.show()

#     gt_background_image = image.copy()

#     for (y, x, k) in data["gt_rpn_background_indices"]:

#         color = (0, 255, 0)

#         height = anchor_map[y,x,k*4+2]
#         width = anchor_map[y,x,k*4+3]
#         cy = anchor_map[y,x,k*4+0]
#         cx = anchor_map[y,x,k*4+1]

#         corners = np.array([cy - 0.5 * height, cx - 0.5 * width, cy + 0.5 * height, cx + 0.5 * width])
#         y1, x1, y2, x2 = corners.astype(int)

#         cv2.rectangle(gt_background_image, (x1, y1), (x2, y2), color, 1)

#     plt.imshow(gt_background_image)
#     plt.show()

#     break

# # anchors = get_anchors(C)