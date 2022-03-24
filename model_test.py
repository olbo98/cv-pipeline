from ast import Mod
import unittest
import os
import tkinter as tk
from model import Module
from view import View
import numpy as np

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
        module = Module(None, "")
        image = module.prepocess_img('girl.png') # (1, 416, 416, 3)
        image_shape = image.shape
        test_array = np.full((1,416,416,3),0)
        test_array_shape = test_array.shape

        self.assertEqual(image_shape, test_array_shape, "Wrong shape")


if __name__ == "__main__":
    unittest.main()