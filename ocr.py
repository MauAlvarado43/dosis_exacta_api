from PIL import Image
import cv2 as cv
import numpy as np
import time

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline()

import easyocr
reader = easyocr.Reader(['es', 'en'], gpu=False)

video = cv.VideoCapture('test.mp4')

if video.isOpened()== False: 
  print("Error opening video stream or file")
 
start_time = time.time()
while video.isOpened():
  
  ret, frame = video.read()

  if ret == True:

    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # img = Image.fromarray(frame_gray)

    # print(pytesseract.image_to_string(img, lang='spa'))
    # print(pipeline.recognize([frame])[0])
    print(reader.readtext(frame))

    cv.imshow('Frame', frame)
    
    if cv.waitKey(25) & 0xFF == ord('q'): break

  else: 
    break

end_time = time.time()
print(f"Time taken: {end_time - start_time}")

video.release()
cv.destroyAllWindows()