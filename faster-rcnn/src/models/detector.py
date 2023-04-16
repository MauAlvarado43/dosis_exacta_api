import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import backend as K
from src.models.roi_pooling import RoIPooling

class DetectorNetwork(Model):

    def __init__(self, C, num_classes, custom_roi_pool, activate_class_outputs, l2, dropout_probability):
        super().__init__()

        self._C = C
        self._num_classes = num_classes
        self._activate_class_outputs = activate_class_outputs
        self._dropout_probability = dropout_probability

        regularizer = tf.keras.regularizers.l2(l2)
        class_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01)
        regressor_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.001)

        self._roi_pool = RoIPooling(pool_size = 7, name = "custom_roi_pool") if custom_roi_pool else None

        # Fully Connected layers with optional dropout
        self._flatten = TimeDistributed(Flatten())
        self._fc1 = TimeDistributed(name = "fc1", layer = Dense(units = 4096, activation = "relu", kernel_regularizer = regularizer))
        self._dropout1 = TimeDistributed(Dropout(dropout_probability))
        self._fc2 = TimeDistributed(name = "fc2", layer = Dense(units = 4096, activation = "relu", kernel_regularizer = regularizer))
        self._dropout2 = TimeDistributed(Dropout(dropout_probability))

        # Output: classifier
        class_activation = "softmax" if activate_class_outputs else None
        self._classifier = TimeDistributed(name = "classifier_class", layer = Dense(units = num_classes, activation = class_activation, kernel_initializer = class_initializer))

        # Output: box delta regressions. Unique regression weights for each possible class excluding background class
        self._regressor = TimeDistributed(name = "classifier_boxes", layer = Dense(units = 4 * (num_classes - 1), activation = "linear", kernel_initializer = regressor_initializer))

    def call(self, inputs, training = False):

        # Unpack inputs
        input_image = inputs[0]
        feature_map = inputs[1]
        proposals = inputs[2]

        assert len(feature_map.shape) == 4, "Feature map must be 4D tensor"

        # RoI Pooling: creates a 7x7 map for each proposal (1, num_rois, 7, 7, 512)

        if self._roi_pool is not None:
            # Use custom RoI Pooling layer, convert (y1, x1, y2, x2) to (y1, x1, height, width)
            proposals = tf.cast(proposals, dtype = tf.float32)
            map_dimensions = tf.shape(feature_map)[1:3]
            map_limits = tf.tile(map_dimensions, multiples = [2]) - 1
            roi_corners = tf.minimum(proposals // 16, map_limits)
            roi_corners = tf.maximum(roi_corners, 0)
            roi_dimensions = roi_corners[:, 2:4] - roi_corners[:, 0:2] + 1
            rois = tf.concat([roi_corners[:, 0:2], roi_dimensions], axis = 1)
            rois = tf.expand_dims(rois, axis = 0)
            pool = RoIPooling(pool_size = 7, name = "roi_pool")([feature_map, rois])
        else:

            # Convert to normalized RoIs with each coordinates in range [0, 1]
            image_height = tf.shape(input_image)[1]
            image_width = tf.shape(input_image)[2]
            rois = proposals / [image_height, image_width, image_height, image_width]

            # Crop, resize, pool
            num_rois = tf.shape(rois)[0]      

            region = tf.image.crop_and_resize(image = feature_map, boxes = rois, box_indices = tf.zeros(num_rois, dtype = tf.int32), crop_size = [14, 14])
            pool = tf.nn.max_pool(region, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
            pool = tf.expand_dims(pool, axis = 0) # (num_rois, 7, 7, 512) -> (1, num_rois, 7, 7, 512)

        # Pass through fully connected layers
        flattened = self._flatten(pool)

        if training and self._dropout_probability != 0:
            fc1 = self._fc1(flattened)
            do1 = self._dropout1(fc1)
            fc2 = self._fc2(do1)
            do2 = self._dropout2(fc2)
            out = do2
        else:
            fc1 = self._fc1(flattened)
            fc2 = self._fc2(fc1)
            out = fc2

        class_activation = "softmax" if self._activate_class_outputs else None
        classes = self._classifier(out)
        box_deltas = self._regressor(out)

        return [classes, box_deltas]
    
    @staticmethod
    def class_loss(y_predicted, y_true, from_logits):
        scale_factor = 1.0
        N = tf.cast(tf.shape(y_true)[1], dtype = tf.float32) + K.epsilon() # Number of ROIs
        if from_logits:
            return scale_factor * K.sum(K.categorical_crossentropy(target = y_true, output = y_predicted, from_logits = True)) / N
        else:
            return scale_factor * K.sum(K.categorical_crossentropy(y_true, y_predicted)) / N
        
    @staticmethod
    def regression_loss(y_predicted, y_true):

        scale_factor = 1.0
        sigma = 1.0
        sigma_squared = sigma * sigma

        # We want to unpack the resession targets and the mask of valid targets into tensors
        # each of the same shape as the predicted (batch_size, num_proposals, 4 * (num_classes - 1))
        # and true (batch_size, num_proposals, 2, 4 * (num_classes - 1))
        y_mask = y_true[:, :, 0, :]
        y_true_targets = y_true[:, :, 1, :]

        # Compute element-wise loss using robust L1 function for all 4 regression targets
        x = y_true_targets - y_predicted
        x_abs = tf.math.abs(x)
        is_negative_branch = tf.stop_gradient(tf.cast(tf.less(x_abs, 1.0 / sigma_squared), dtype = tf.float32))
        R_negative_branch = 0.5 * x * x * sigma_squared
        R_positive_branch = x_abs - 0.5 / sigma_squared
        losses = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

        # Accumulate the relevant terms and normalize by the number of proposals
        N = tf.cast(tf.shape(y_true)[1], dtype = tf.float32) + K.epsilon()  # N = number of proposals
        relevant_loss_terms = y_mask * losses
        return scale_factor * K.sum(relevant_loss_terms) / N