from __future__ import annotations
import os
import unittest
import math
from module import Module
from pool import Pool

class TestDistFromPoint(unittest.TestCase):
    def test_dist_from_point(self):
        module = Module(None, '', {}, {})
        box = [0.0, 0.0, 4.0, 4.0]
        annotation = (3.0, 1.0)
        self.assertEqual(module.dist_from_point(box, annotation), math.sqrt(2), "Should be square root of 2")

class TestFindClosestBox(unittest.TestCase):
    def test_find_closes_box(self):
        module = Module(None, '', {}, {})
        boxes = [[0.0, 0.0, 4.0, 4.0], [6.0, 5.0, 8.0, 7.0], [3.0, 1.0, 7.0, 5.0]]
        scores = [0,98, 0.99, 0.95]
        classes = [1, 2, 3]
        annotation = (4.0, 4.0)
        self.assertEqual(module.find_closest_box(boxes, scores, classes, annotation), (boxes[2], scores[2], classes[2]), "Should be ([3.0, 1.0, 7.0, 5.0], 0.95, 3)")

class TestPseudoLabels(unittest.TestCase):
    def test_pseudo_labels(self):
        module = Module(None, 'D:/Exjobb/yolov3-tf2/data/', {}, {})
        model = module.setup_model()
        sample = ['girl.png']
        weak_annotations = [[[406,333], [0,0]]]
        p_s = module.pseudo_labels(model, sample, weak_annotations)
        num_boxes = len(p_s[0][0])
        self.assertEqual(num_boxes, len(weak_annotations[0]), "Number of bounding boxes should be equal to the number of weak annotations")

class TestActiveSampling(unittest.TestCase):
    def test_active_sampling(self):
        path = 'D:/Exjobb/yolov3-tf2/data/'
        module = Module(None, path, {}, {})
        model = module.setup_model()
        unlabeled_pool = ['girl.png', 'street.jpg']
        lowest_score = float('inf')
        uncertain_image = ''
        for image in unlabeled_pool:
            img = module.prepocess_img(os.path.join(path, image))
            _, scores, _, _ = model.predict(img)
            if scores[0][0] < lowest_score:
                lowest_score = scores[0][0]
                uncertain_image = image
                
        samples = module.active_smapling(model, unlabeled_pool, 1)
        self.assertEqual(uncertain_image, samples[0], "The images should be the one with the lowest score")

if __name__ == "__main__":
    unittest.main()