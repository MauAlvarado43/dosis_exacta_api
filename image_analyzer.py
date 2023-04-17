import cv2 as cv
import torch
import pytesseract
import spacy

model_name = "best.pt"
model = torch.hub.load("WongKinYiu/yolov7", 'custom', model_name)

nlp = spacy.load("es_core_news_sm")

def get_medical_region(image):

    model.eval()
    results = model(image, size = 640)
    
    bboxes = results.pandas().xyxy[0].values
    best_bbox = bboxes[0]
    medical_region = image[int(best_bbox[1]) : int(best_bbox[3]), int(best_bbox[0]) : int(best_bbox[2])]

    return medical_region

def analyze_image(image):
    medical_region = get_medical_region(image)
    text = pytesseract.image_to_string(medical_region)
    return text

def clean_text(corpus):

    new_corpus = []
    temp = ""

    corpus = corpus.lower()

    to_replace = ["!", "?", ".", ",", ":", ";", "(", ")", "[", "]", "{", "}", "/", "\\", "|", "°", "º", "ª", "`", "¨", "~", "-", "_"]
    for char in to_replace: corpus = corpus.replace(char, "")

    for text in corpus.split("\n"):
        if text.strip() != "":
            temp += text + " "
        else:
            new_corpus.append(temp)
            temp = ""

    if temp != "":
        new_corpus.append(temp)

    normalized_text = []
    stop_words = ["ADP", "CONJ", "CCONJ", "PRON", "SYM", "PUNCT"]
    new_corpus = [nlp(text) for text in new_corpus]

    for text in new_corpus:

        temp = []

        for token in text:
            if token.pos_ not in stop_words:
                temp.append(token.text)

        normalized_text.append(temp)

    normalized_text = [text for text in normalized_text if text != []]

    return normalized_text


def get_indications(image):

    # image = cv.imdecode(image, cv.IMREAD_COLOR)
    text = analyze_image(image)
    text = clean_text(text)

    print(text)

    return [{

    }]

if __name__ == "__main__":
    image = cv.imread("images/0.jpg")
    get_indications(image)