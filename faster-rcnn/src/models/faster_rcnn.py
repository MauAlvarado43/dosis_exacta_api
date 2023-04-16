import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from src.models.vgg import FeatureExtractor
from src.models.rpn import RegionProposalNetwork
from src.models.detector import DetectorNetwork
from src.data.math import convert_deltas_to_boxes, tf_intersection_over_union

class FasterRCNNModel(Model):

    def __init__(self, C, num_classes, allow_edge_proposals, custom_roi_pool, activate_class_outputs, l2 = 0, dropout_probability = 0):
        super().__init__()

        self.C = C
        self._num_classes = num_classes
        self._activate_class_outputs = activate_class_outputs
        self._stage1_feature_extractor = FeatureExtractor(C, l2 = l2)
        self._stage2_region_proposal_network = RegionProposalNetwork(
            C,
            max_proposals_pre_nms_train = 12000,
            max_proposals_post_nms_train = 2000,
            max_proposals_pre_nms_infer = 6000,
            max_proposals_post_nms_infer = 300,
            l2 = l2,
            allow_edge_proposals = allow_edge_proposals
        )
        self._stage3_detector_network = DetectorNetwork(
            C,
            num_classes = num_classes,
            custom_roi_pool = custom_roi_pool,
            activate_class_outputs = activate_class_outputs,
            l2 = l2,
            dropout_probability = dropout_probability
        )

    def call(self, inputs, training = False):

        # Unpack inputs
        input_image = inputs[0] # (1, height_pixels, width_pixels, 3)
        anchor_map = inputs[1] # (1, height, width, num_anchors * 4)
        anchor_valid_map = inputs[2] # (1, height, width, num_anchors)

        if training:
            gt_rpn_map = inputs[3] # (1, height, width, num_anchors, 6)
            gt_box_class_idxs_map = inputs[4] # (1, num_gt_boxes)
            gt_box_corners_map = inputs[5] # (1, num_gt_boxes, 4)

        # Stage 1: Extract features
        feature_map = self._stage1_feature_extractor(input_image = input_image, training = training)

        # Stage 2: Generate object proposals using RPN
        rpn_scores, rpn_box_deltas, proposals = self._stage2_region_proposal_network(
            inputs = [
                input_image,
                feature_map,
                anchor_map,
                anchor_valid_map
            ],
            training = training
        )

        # If training, we must generate GT data for the detector stage from RPN outputs
        if training:

            # Assign labels to proposals and take random sample (for detector training)
            proposals, gt_classes, gt_box_deltas = self._label_proposals(
                proposals = proposals,
                gt_box_class_idxs = gt_box_class_idxs_map[0], # for now, batch size of 1
                gt_box_corners = gt_box_corners_map[0],
                min_background_iou_threshold = 0.0,
                min_object_iou_threshold = 0.5
            )
            proposals, gt_classes, gt_box_deltas = self._sample_proposals(
                proposals = proposals,
                gt_classes = gt_classes,
                gt_box_deltas = gt_box_deltas,
                max_proposals = 128,
                positive_fraction = 0.25
            )
            gt_classes = tf.expand_dims(gt_classes, axis = 0) # (N,num_classes) -> (1,N,num_classes) (as expected by loss function)
            gt_box_deltas = tf.expand_dims(gt_box_deltas, axis = 0) # (N,2,(num_classes-1)*4) -> (1,N,2,(num_classes-1)*4)

            # Ensure proposals are treated as constants and do not propagate gradients
            proposals = tf.stop_gradient(proposals)
            gt_classes = tf.stop_gradient(gt_classes)
            gt_box_deltas = tf.stop_gradient(gt_box_deltas)

        # Stage 3: Detector
        detector_classes, detector_box_deltas = self._stage3_detector_network(
            inputs = [
                input_image,
                feature_map,
                proposals
            ],
            training = training
        )

        # Losses

        if training:

            rpn_class_loss = self._stage2_region_proposal_network.class_loss(y_predicted = rpn_scores, gt_rpn_map = gt_rpn_map)
            rpn_regression_loss = self._stage2_region_proposal_network.regression_loss(y_predicted = rpn_box_deltas, gt_rpn_map = gt_rpn_map)
            detector_class_loss = self._stage3_detector_network.class_loss(y_predicted = detector_classes, y_true = gt_classes, from_logits = not self._activate_class_outputs)
            detector_regression_loss = self._stage3_detector_network.regression_loss(y_predicted = detector_box_deltas, y_true = gt_box_deltas)
           
            self.add_loss(rpn_class_loss)
            self.add_loss(rpn_regression_loss)
            self.add_loss(detector_class_loss)
            self.add_loss(detector_regression_loss)

            self.add_metric(rpn_class_loss, name = "rpn_class_loss")
            self.add_metric(rpn_regression_loss, name = "rpn_regression_loss")
            self.add_metric(detector_class_loss, name = "detector_class_loss")
            self.add_metric(detector_regression_loss, name = "detector_regression_loss")

        else:
        
            # Losses cannot be computed during inference and should be ignored
            rpn_class_loss = float("inf")
            rpn_regression_loss = float("inf")
            detector_class_loss = float("inf")
            detector_regression_loss = float("inf")

        # Return outputs
        return [
            rpn_scores,
            rpn_box_deltas,
            detector_classes,
            detector_box_deltas,
            proposals,
            rpn_class_loss,
            rpn_regression_loss,
            detector_class_loss,
            detector_regression_loss
        ]
    
    # Run inference on a batch of images.
    # x: list of input maps (input image, anchor map, anchor valid map)
    # score_threshold: minimum score for a box to be considered a positive detection
    def predict_on_batch(self, x, score_threshold):
        _, _, detector_classes, detector_box_deltas, proposals, _, _, _, _ = super().predict_on_batch(x = x)
        scored_boxes_by_class_index = self._predictions_to_scored_boxes(
            input_image = x[0],
            classes = detector_classes,
            box_deltas = detector_box_deltas,
            proposals = proposals,
            score_threshold = score_threshold
        )
        return scored_boxes_by_class_index
    
    # Load weights from Keras VGG-16 model pre-trained on ImageNet into the
    # feature extractor convolutional layers as well as the two fully connected
    # layers in the detector stage.
    def load_imagenet_weights(self):

        keras_model = tf.keras.applications.VGG16(weights = "imagenet")

        for keras_layer in keras_model.layers:

            weights = keras_layer.get_weights()

            if len(weights) > 0:

                vgg16_layers = self._stage1_feature_extractor.layers + self._stage3_detector_network.layers
                our_layer = [ layer for layer in vgg16_layers if layer.name == keras_layer.name ]

                if len(our_layer) > 0:
                    print("Loading VGG-16 ImageNet weights into layer: %s" % our_layer[0].name)
                    our_layer[0].set_weights(weights)

    def _predictions_to_scored_boxes(self, input_image, classes, box_deltas, proposals, score_threshold):

        # Eliminate batch dimension
        input_image = np.squeeze(input_image, axis = 0)
        classes = np.squeeze(classes, axis = 0)
        box_deltas = np.squeeze(box_deltas, axis = 0)

        # Convert logits to probability distribution if using logits mode
        if not self._activate_class_outputs:
            classes = tf.nn.softmax(classes, axis = 1).numpy()

        # Convert proposal boxes -> center point and size
        proposal_anchors = np.empty(proposals.shape)
        proposal_anchors[:, 0] = 0.5 * (proposals[:, 0] + proposals[:, 2]) # center_y
        proposal_anchors[:, 1] = 0.5 * (proposals[:, 1] + proposals[:, 3]) # center_x
        proposal_anchors[:, 2:4] = proposals[:, 2:4] - proposals[:, 0:2] # height, width

        # Separate out results per class: class_idx -> (y1, x1, y2, x2, score)
        boxes_and_scores_by_class_idx = {}

        for class_idx in range(1, classes.shape[1]):  # skip class 0 (background)

            # Get the regression parameters (ty, tx, th, tw) corresponding to this
            # class, for all proposals
            box_delta_idx = (class_idx - 1) * 4
            box_delta_params = box_deltas[:, (box_delta_idx + 0) : (box_delta_idx + 4)] # (N, 4)
            proposal_boxes_this_class = convert_deltas_to_boxes(
                box_deltas = box_delta_params,
                anchors = proposal_anchors,
                box_delta_means = [0.0, 0.0, 0.0, 0.0],
                box_delta_stds = [0.1, 0.1, 0.2, 0.2]
            )

            # Clip to image boundaries
            proposal_boxes_this_class[:, 0::2] = np.clip(proposal_boxes_this_class[:, 0::2], 0, input_image.shape[0] - 1) # clip y1 and y2 to [0, height)
            proposal_boxes_this_class[:, 1::2] = np.clip(proposal_boxes_this_class[:, 1::2], 0, input_image.shape[1] - 1) # clip x1 and x2 to [0, width)

            # Get the scores for this class. The class scores are returned in
            # normalized categorical form. Each row corresponds to a class.
            scores_this_class = classes[:,class_idx]

            # Keep only those scoring high enough
            sufficiently_scoring_idxs = np.where(scores_this_class > score_threshold)[0]
            proposal_boxes_this_class = proposal_boxes_this_class[sufficiently_scoring_idxs]
            scores_this_class = scores_this_class[sufficiently_scoring_idxs]
            boxes_and_scores_by_class_idx[class_idx] = (proposal_boxes_this_class, scores_this_class)

        # Perform NMS per class
        scored_boxes_by_class_idx = {}
        for class_idx, (boxes, scores) in boxes_and_scores_by_class_idx.items():
            idxs = tf.image.non_max_suppression(boxes = boxes, scores = scores, max_output_size = proposals.shape[0], iou_threshold = 0.3)
            idxs = idxs.numpy()
            boxes = boxes[idxs]
            scores = np.expand_dims(scores[idxs], axis = 0) # (N,) -> (N,1)
            scored_boxes = np.hstack([ boxes, scores.T ]) # (N,5), with each row: (y1, x1, y2, x2, score)
            scored_boxes_by_class_idx[class_idx] = scored_boxes

        return scored_boxes_by_class_idx
    
    # Determines which proposals generated by the RPN stage overlap with ground
    # truth boxes and creates ground truth labels for the subsequent detector stage.
    # proposals: proposal corners (y_min, x_min, y_max, x_max)
    # gt_box_class_idxs: The class index for each ground truth box, shaped (M, ), where M is the number of ground truth boxes.
    # gt_box_corners: Ground truth box corners, shaped (M, 4).
    # min_background_iou_threshold: Minimum IoU threshold with ground truth boxes
    # min_object_iou_threshold: Minimum IoU threshold for a proposal to be labeled as an object.
    def _label_proposals(self, proposals, gt_box_class_idxs, gt_box_corners, min_background_iou_threshold, min_object_iou_threshold):

        # Let's be crafty and create some fake proposals that match the ground truth boxes exactly. This isn't strictly necessary and
        # the model should work without it but it will help training and will ensure that there are always some positive examples to train on.
        proposals = tf.concat([ proposals, gt_box_corners ], axis = 0)

        # Compute IoU between each proposal (N,4) and each ground truth box (M,4) -> (N, M)
        ious = tf_intersection_over_union(boxes1 = proposals, boxes2 = gt_box_corners)

        # Find the best IoU for each proposal, the class of the ground truth box associated with it, and the box corners
        best_ious = tf.math.reduce_max(ious, axis = 1) # (N, ) of maximum IoUs for each of the N proposals
        box_idxs = tf.math.argmax(ious, axis = 1) # (N, ) of ground truth box index for each proposal
        gt_box_class_idxs = tf.gather(gt_box_class_idxs, indices = box_idxs) # (N, ) of class indices of highest-IoU box for each proposal
        gt_box_corners = tf.gather(gt_box_corners, indices = box_idxs) # (N, 4) of box corners of highest-IoU box for each proposal

        # Remove all proposals whose best IoU is less than the minimum threshold
        # for a negative (background) sample. We also check for IoUs > 0 because
        # due to earlier clipping, we may get invalid 0-area proposals.
        idxs = tf.where(best_ious >= min_background_iou_threshold)  # keep proposals w/ sufficiently high IoU
        proposals = tf.gather_nd(proposals, indices = idxs)
        best_ious = tf.gather_nd(best_ious, indices = idxs)
        gt_box_class_idxs = tf.gather_nd(gt_box_class_idxs, indices = idxs)
        gt_box_corners = tf.gather_nd(gt_box_corners, indices = idxs)

        # IoUs less than min_object_iou_threshold will be labeled as background
        retain_mask = tf.cast(best_ious >= min_object_iou_threshold, dtype = gt_box_class_idxs.dtype) # (N, ), with 0 wherever best_iou < threshold, else 1
        gt_box_class_idxs = gt_box_class_idxs * retain_mask

        # One-hot encode class labels
        num_classes = self._num_classes
        gt_classes = tf.one_hot(indices = gt_box_class_idxs, depth = num_classes) # (N,num_classes)

        # Convert proposals and ground truth boxes into "anchor" format (center points and side lengths). For the detector stage, 
        # the proposals serve as the anchors relative to which the final box predictions will be regressed.
        proposal_centers = 0.5 * (proposals[:,0:2] + proposals[:,2:4]) # center_y, center_x
        proposal_sides = proposals[:,2:4] - proposals[:,0:2] # height, width
        gt_box_centers = 0.5 * (gt_box_corners[:,0:2] + gt_box_corners[:,2:4]) # center_y, center_x
        gt_box_sides = gt_box_corners[:,2:4] - gt_box_corners[:,0:2] # height, width

        # Compute regression targets (ty, tx, th, tw) for each proposal based on the best box selected
        detector_box_delta_means = tf.constant([0, 0, 0, 0], dtype = tf.float32)
        detector_box_delta_stds = tf.constant([0.1, 0.1, 0.2, 0.2], dtype = tf.float32)
        tyx = (gt_box_centers - proposal_centers) / proposal_sides  # ty = (gt_center_y - proposal_center_y) / proposal_height, tx = (gt_center_x - proposal_center_x) / proposal_width
        thw = tf.math.log(gt_box_sides / proposal_sides) # th = log(gt_height / proposal_height), tw = (gt_width / proposal_width)
        box_delta_targets = tf.concat([ tyx, thw ], axis = 1) # (N,4) box delta regression targets tensor
        box_delta_targets = (box_delta_targets - detector_box_delta_means) / detector_box_delta_stds # mean and standard deviation adjustment

        # Convert regression targets into a map of shape (N, 2, 4 * (C - 1)) where C is the number of classes and [:, 0, :] specifies a mask for the corresponding
        # target components at [:, 1, :]. Targets are ordered (ty, tx, th, tw). Background class 0 is not present at all.
        gt_box_deltas_mask = tf.repeat(gt_classes, repeats = 4, axis = 1)[:,4:] # create masks using interleaved repetition, remembering to discard class 0
        gt_box_deltas_values = tf.tile(box_delta_targets, multiples = [1, num_classes - 1]) # populate regression targets with straightforward repetition of each row (only those columns corresponding to class will be masked on)
        gt_box_deltas_mask = tf.expand_dims(gt_box_deltas_mask, axis = 0) # (N, 4 * (C - 1)) -> (1, N, 4 * (C - 1))
        gt_box_deltas_values = tf.expand_dims(gt_box_deltas_values, axis = 0) # (N, 4 * (C - 1)) -> (1, N, 4 * (C - 1))
        gt_box_deltas = tf.concat([ gt_box_deltas_mask, gt_box_deltas_values ], axis = 0) # (2, N, 4 * (C - 1))
        gt_box_deltas = tf.transpose(gt_box_deltas, perm = [ 1, 0, 2]) # (N, 2, 4 * (C - 1))

        return proposals, gt_classes, gt_box_deltas

    def _sample_proposals(self, proposals, gt_classes, gt_box_deltas, max_proposals, positive_fraction):

        if max_proposals <= 0:
            return proposals, gt_classes, gt_box_deltas

        # Get positive and negative (background) proposals
        class_indices = tf.argmax(gt_classes, axis = 1) # (N, num_classes) -> (N, ), where each element is the class index (highest score from its row)
        positive_indices = tf.squeeze(tf.where(class_indices > 0), axis = 1)  # (P, ), tensor of P indices (the positive, non-background classes in class_indices)
        negative_indices = tf.squeeze(tf.where(class_indices <= 0), axis = 1) # (N, ), tensor of N indices (the negative, background classes in class_indices)
        num_positive_proposals = tf.size(positive_indices)
        num_negative_proposals = tf.size(negative_indices)

        # Select positive and negative samples, if there are enough.
        num_samples = tf.minimum(max_proposals, tf.size(class_indices))
        num_positive_samples = tf.minimum(tf.cast(tf.math.round(tf.cast(num_samples, dtype = float) * positive_fraction), dtype = num_samples.dtype), num_positive_proposals)
        num_negative_samples = tf.minimum(num_samples - num_positive_samples, num_negative_proposals)

        # Sample randomly
        positive_sample_indices = tf.random.shuffle(positive_indices)[:num_positive_samples]
        negative_sample_indices = tf.random.shuffle(negative_indices)[:num_negative_samples]
        indices = tf.concat([ positive_sample_indices, negative_sample_indices ], axis = 0)

        return tf.gather(proposals, indices = indices), tf.gather(gt_classes, indices = indices), tf.gather(gt_box_deltas, indices = indices)