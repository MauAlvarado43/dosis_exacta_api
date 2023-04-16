import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.initializers import glorot_normal

class FeatureExtractor(Model):

    def __init__(self, C, l2 = 0):
        super().__init__()

        self._C = C
        initial_weights = glorot_normal()
        regularizer = tf.keras.regularizers.l2(l2)
        input_shape = (None, None, 3)

        self._block1_conv1 = Conv2D(name = "block1_conv1", kernel_size = (3, 3), strides = 1, filters = 64, padding = "same", activation = "relu", kernel_initializer = initial_weights, trainable = False, input_shape = input_shape)
        self._block1_conv2 = Conv2D(name = "block1_conv2", kernel_size = (3, 3), strides = 1, filters = 64, padding = "same", activation = "relu", kernel_initializer = initial_weights, trainable = False)
        self._block1_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

        self._block2_conv1 = Conv2D(name = "block2_conv1", kernel_size = (3, 3), strides = 1, filters = 128, padding = "same", activation = "relu", kernel_initializer = initial_weights, trainable = False)
        self._block2_conv2 = Conv2D(name = "block2_conv2", kernel_size = (3, 3), strides = 1, filters = 128, padding = "same", activation = "relu", kernel_initializer = initial_weights, trainable = False)
        self._block2_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

        self._block3_conv1 = Conv2D(name = "block3_conv1", kernel_size = (3, 3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self._block3_conv2 = Conv2D(name = "block3_conv2", kernel_size = (3, 3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self._block3_conv3 = Conv2D(name = "block3_conv3", kernel_size = (3, 3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self._block3_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

        self._block4_conv1 = Conv2D(name = "block4_conv1", kernel_size = (3, 3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self._block4_conv2 = Conv2D(name = "block4_conv2", kernel_size = (3, 3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self._block4_conv3 = Conv2D(name = "block4_conv3", kernel_size = (3, 3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self._block4_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

        self._block5_conv1 = Conv2D(name = "block5_conv1", kernel_size = (3, 3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self._block5_conv2 = Conv2D(name = "block5_conv2", kernel_size = (3, 3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self._block5_conv3 = Conv2D(name = "block5_conv3", kernel_size = (3, 3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)

    def call(self, input_image):

        y = self._block1_conv1(input_image)
        y = self._block1_conv2(y)
        y = self._block1_maxpool(y)

        y = self._block2_conv1(y)
        y = self._block2_conv2(y)
        y = self._block2_maxpool(y)

        y = self._block3_conv1(y)
        y = self._block3_conv2(y)
        y = self._block3_conv3(y)
        y = self._block3_maxpool(y)

        y = self._block4_conv1(y)
        y = self._block4_conv2(y)
        y = self._block4_conv3(y)
        y = self._block4_maxpool(y)

        y = self._block5_conv1(y)
        y = self._block5_conv2(y)
        y = self._block5_conv3(y)

        return y