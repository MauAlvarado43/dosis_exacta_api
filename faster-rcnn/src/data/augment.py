import cv2
import numpy as np

def apply_augment(img, gt_boxes, C, augment = True):

    augmentations = []
    height, width, _ = img.shape

    if augment:     

        if C.use_horizontal_flips and np.random.randint(0, 2) == 0:
            
            img = cv2.flip(img, 1)
            augmentations.append("horizontal_flip")

            for gt_box in gt_boxes:
                y1, x1, y2, x2 = gt_box["corners"]
                gt_box["corners"] = np.array([y1, width - x2, y2, width - x1])

        if C.use_vertical_flips and np.random.randint(0, 2) == 0:

            img = cv2.flip(img, 0)
            augmentations.append("vertical_flip")

            for gt_box in gt_boxes:
                y1, x1, y2, x2 = gt_box["corners"]
                gt_box["corners"] = np.array([height - y2, x1, height - y1, x2])

        # if C.use_rot_90 and np.random.randint(0, 2) == 0:
            
        #     img = np.rot90(img)
        #     augmentations.append("rot_90")

        #     for gt_box in gt_boxes:
        #         y1, x1, y2, x2 = gt_box["corners"]
        #         gt_box["corners"] = np.array([x1, height - y2, x2, height - y1])

        if len(augmentations) == 0:
            augmentations.append("original")

    else:
            
        augmentations = ["original"]

    return img, gt_boxes, augmentations