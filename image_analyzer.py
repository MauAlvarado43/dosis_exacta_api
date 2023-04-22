import cv2 as cv
import torch
import pytesseract
import numpy as np
import base64
from text_analyzer import extract_gtp_indications

model_name = "best.pt"
model = torch.hub.load("WongKinYiu/yolov7", 'custom', model_name)

def get_medical_region(image):

    model.eval()
    results = model(image, size = 640)
    
    bboxes = results.pandas().xyxy[0].values
    best_bbox = bboxes[0]
    medical_region = image[int(best_bbox[1]) : int(best_bbox[3]), int(best_bbox[0]) : int(best_bbox[2])]

    return medical_region

def analyze_image(image):

    medical_region = get_medical_region(image)

    cv.imwrite("./temp/medical_region.jpg", medical_region)
    cv.imwrite("./temp/medical_receipe.jpg", image)

    text = pytesseract.image_to_string(medical_region)

    with open("temp/text.txt", "w", encoding = "utf-8") as f:
        f.write(text)

    return text

def get_indications(form_image):
    
    image = base64.b64decode(form_image)
    image = np.frombuffer(image, dtype = np.uint8)
    image = cv.imdecode(image, cv.IMREAD_COLOR)

    text = analyze_image(image)
    indications = extract_gtp_indications(text)

    return indications

if __name__ == "__main__":
    image = cv.imread("images/2.jpg")
    get_indications(image)