import cv2 as cv
import keras_ocr

pipeline = keras_ocr.pipeline.Pipeline()

def recognize(image):
    return pipeline.recognize([image])[0]

# frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# print(pipeline.recognize([frame])[0])