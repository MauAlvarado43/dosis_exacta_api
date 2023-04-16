import glob
import numpy as np
import os
import random
import xml.etree.ElementTree as ET
from .image import load_image
from .augment import apply_augment
from .anchors import generate_anchor_maps, generate_rpn_map

class Dataset:

    def __init__(self, C, dataset_path, shuffle = True, cache = True):

        if not os.path.exists(dataset_path):
            raise Exception("Dataset path does not exist")
        
        self.C = C
        self.dataset_path = dataset_path
        self.augment = self.C.augment
        self.shuffle = shuffle
        self.cache = cache

        self.class_index_to_name = self.get_classes()
        self.class_name_to_index = { class_name: class_index for (class_index, class_name) in self.class_index_to_name.items() }
        self.filepaths = self.get_filepaths()
        self.gt_boxes = self.get_ground_truth_boxes(self.filepaths)
        self.num_classes = len(self.class_index_to_name)
        self.cache_images = {}
        self.i = 0
        self.num_samples = len(self.filepaths)

    def __iter__(self):
        self.i = 0
        if self.shuffle: random.shuffle(self.filepaths)
        return self
    
    def __next__(self):

        if self.i >= len(self.filepaths):
            raise StopIteration

        filepath = self.filepaths[self.i]
        self.i += 1

        return self.get_sample(filepath)

    def get_sample(self, filepath):

        image, scale_factor, _ = load_image(filepath, max_size = self.C.max_image_size)

        # Scale GT boxes
        scaled_gt_boxes = []
        for gt_box in self.gt_boxes[filepath]:
            corners = gt_box["corners"]
            scaled_gt_box = {
                "corners": corners * scale_factor,
                "class_name": gt_box["class_name"],
                "class_index": gt_box["class_index"]
            }
            scaled_gt_boxes.append(scaled_gt_box)
            
        # Check augmentations
        image, gt_boxes, augmentations = apply_augment(image, scaled_gt_boxes, self.C, augment = self.augment)
        applied_augmentations = ".".join(augmentations)
        if filepath in self.cache_images and self.cache_images[filepath]["augmentations"] == applied_augmentations and self.cache: return self.cache_images[filepath]

        # Get valid anchors
        anchor_map, anchor_valid_map = generate_anchor_maps(self.C, image)
        gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices = generate_rpn_map(anchor_map = anchor_map, anchor_valid_map = anchor_valid_map, gt_boxes = scaled_gt_boxes)

        data = {
            "image": image,
            "gt_boxes": gt_boxes,
            "anchor_map": anchor_map,
            "anchor_valid_map": anchor_valid_map,
            "gt_rpn_map": gt_rpn_map,
            "gt_rpn_object_indices": gt_rpn_object_indices,
            "gt_rpn_background_indices": gt_rpn_background_indices,
            "augmentations": applied_augmentations,
            "filepath": filepath
        }

        if self.cache: self.cache_images[filepath] = data
        
        return data

    def get_classes(self):

        classes_path = os.path.join(self.dataset_path, "../classes.txt").replace("\\", "/")

        if not os.path.exists(classes_path):
            raise Exception("No classes.txt file found in dataset path")

        class_file = classes_path
        
        with open(class_file) as fp:
            classes = [ line.strip() for line in fp.readlines() ]
        
        assert len(classes) > 0, "No classes found in classes.txt file"

        class_index_to_name = {0: "background"}
        for v in enumerate(sorted(classes)): class_index_to_name[v[0] + 1] = v[1]

        return class_index_to_name
    
    def get_filepaths(self):
        filepaths = glob.glob(self.dataset_path + "/*.jpg")
        assert len(filepaths) > 0, "No image files found in dataset path"
        return [ filepath.replace("\\", "/") for filepath in filepaths ]
    
    def get_ground_truth_boxes(self, filepaths):

        gt_boxes_by_filepath = {}

        for filepath in filepaths:

            basename = os.path.basename(filepath).replace(".jpg", "")
            annotation_file = (os.path.join(self.dataset_path, basename) + ".xml").replace("\\", "/")

            assert os.path.exists(annotation_file), "No annotation file found for %s" % filepath

            tree = ET.parse(annotation_file)
            root = tree.getroot()

            assert tree != None, "Failed to parse %s" % annotation_file
            assert len(root.findall("size")) == 1

            size = root.find("size")

            assert len(size.findall("depth")) == 1

            depth = int(size.find("depth").text)

            assert depth == 3

            boxes = []

            for obj in root.findall("object"):

                assert len(obj.findall("name")) == 1
                assert len(obj.findall("bndbox")) == 1
                assert len(obj.findall("difficult")) == 1
                
                class_name = obj.find("name").text
                bndbox = obj.find("bndbox")

                assert len(bndbox.findall("xmin")) == 1
                assert len(bndbox.findall("ymin")) == 1
                assert len(bndbox.findall("xmax")) == 1
                assert len(bndbox.findall("ymax")) == 1

                x_min = int(bndbox.find("xmin").text) - 1  # convert to 0-based pixel coordinates
                y_min = int(bndbox.find("ymin").text) - 1
                x_max = int(bndbox.find("xmax").text) - 1
                y_max = int(bndbox.find("ymax").text) - 1

                corners = np.array([ y_min, x_min, y_max, x_max ]).astype(np.float32)
                box = {
                    "class_index": self.class_name_to_index[class_name],
                    "class_name": class_name,
                    "corners": corners
                }
                boxes.append(box)

            assert len(boxes) > 0

            gt_boxes_by_filepath[filepath] = boxes
        
        return gt_boxes_by_filepath 