from __future__ import annotations
import os
import sys

sys.path.append('D:/Exjobb/cv-pipeline') 

import unittest
import math
from module import Module
from view import View
import tkinter as tk
from pool import Pool
import numpy as np

class TestDistFromPoint(unittest.TestCase):
    def test_dist_from_point(self):
        module = Module(None, '')
        box = [0.0, 0.0, 4.0, 4.0]
        annotation = (3.0, 1.0)
        self.assertEqual(module.dist_from_point(box, annotation), math.sqrt(2), "Should be square root of 2")

class TestFindClosestBox(unittest.TestCase):
    def test_find_closes_box(self):
        module = Module(None, '')
        boxes = [[0.0, 0.0, 4.0, 4.0], [6.0, 5.0, 8.0, 7.0], [3.0, 1.0, 7.0, 5.0]]
        scores = [0,98, 0.99, 0.95]
        classes = [1, 2, 3]
        annotation = (4.0, 4.0)
        self.assertEqual(module.find_closest_box(boxes, scores, classes, annotation), (boxes[2], scores[2], classes[2]), "Should be ([3.0, 1.0, 7.0, 5.0], 0.95, 3)")

class TestPseudoLabels(unittest.TestCase):
    def test_pseudo_labels(self):
        module = Module(None, './tests/images')
        model = module.setup_model()
        sample = ['girl.png']
        weak_annotations = [[[406,333], [0,0]]]
        p_s = module.pseudo_labels(model, sample, weak_annotations)
        num_boxes = len(p_s[0][0])
        self.assertEqual(num_boxes, len(weak_annotations[0]), "Number of bounding boxes should be equal to the number of weak annotations")

class TestActiveSampling(unittest.TestCase):
    def test_active_sampling(self):
        path = './tests/images'
        module = Module(None, path)
        model = module.setup_model()
        unlabeled_pool = ['girl.png', 'street.jpg']
        lowest_score = float('inf')
        uncertain_image = ''
        for image in unlabeled_pool:
            img = module.prepocess_img(image)
            _, scores, _, _ = model.predict(img)
            if scores[0][0] < lowest_score:
                lowest_score = scores[0][0]
                uncertain_image = image
                
        samples = module.active_smapling(model, unlabeled_pool, 1)
        self.assertEqual(uncertain_image, samples[0], "The images should be the one with the lowest score")

class TestAddGetMethods(unittest.TestCase):

    def test_add_circle_coords(self):
        set_images = ['test.png']
        module = Module(None, "")
        module.prepare_imgs(set_images)
        module.active_image = 'test.png'

        module.add_circle_coords(1,2)
        circle_coords = module.get_circle_coords()
        
        self.assertEqual(circle_coords[module.active_image], [[1,2]], "Does not match the image coordinates" )

    def test_add_rect_coords(self):
        set_images = ['test.png']
        module = Module(None, "")
        module.prepare_imgs(set_images)

        module.active_image = 'test.png'
        module.add_rect_coords(1,2,3,4)
        rect_coords = module.get_rect_coords()

        self.assertEqual(rect_coords[module.active_image], [[1,2,3,4]], "Does not match image coordinates")

class TestDeleteAnnotations(unittest.TestCase):
    
    def test_delete_annotations(self):
        window = tk.Tk()
        view = View(window)
        set_images = ['test.png']
        module = Module(view,"")
        module.prepare_imgs(set_images)
        module.active_image = 'test.png'
        module.shape_IDs.append(5)
        module.add_circle_coords(1,2)

        module.strong_annotations = False
        module.delete_annotations()

        test_coords = {}
        test_coords['test.png'] = []
        test_shape_IDs = []

        
        self.assertEqual(module.circle_coords, test_coords, "Did not delete the coordinates")
        self.assertEqual(module.shape_IDs, test_shape_IDs, "Did not delete the annotation on the canvas")

        module.shape_IDs.append(5)
        module.add_rect_coords(1,2,3,4)
        module.strong_annotations = True
        module.delete_annotations()

        self.assertEqual(module.rect_coords, test_coords, "Did not delete the coordinates")

class TestPreprocessImage(unittest.TestCase):

    def test_preprocess_images(self):
        module = Module(None, "./tests/images")
        image = module.prepocess_img('girl.png') # (1, 416, 416, 3)
        image_shape = image.shape
        test_array = np.full((1,416,416,3),0)
        test_array_shape = test_array.shape

        self.assertEqual(image_shape, test_array_shape, "Wrong shape")

if __name__ == "__main__":
    unittest.main()