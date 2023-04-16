import cv2
import numpy as np

def compute_scale_factor(original_height, original_width, max_size):
    if original_width > original_height: scale_factor = max_size / original_width
    else: scale_factor = max_size / original_height
    return scale_factor
    
def resize_image(image, scale_factor):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    resized_image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return resized_image

# RGB -> BGR
def preprocess_image(image):
    image = image.astype(np.float32)
    image[:, :, 0] -= 103.939
    image[:, :, 1] -= 116.779
    image[:, :, 2] -= 123.680
    return image

def load_image(image_path, max_size = 600):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    scale_factor = compute_scale_factor(height, width, max_size)
    image = resize_image(image, scale_factor)
    image = preprocess_image(image)
    return image, scale_factor, (height, width)