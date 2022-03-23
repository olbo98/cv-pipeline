from __future__ import annotations
import unittest
import math
from module import Module

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
        module = Module(None, '', {}, {})
        model = module.setup_model()
        sample = ['D:/Exjobb/yolov3-tf2/data/girl.png']
        weak_annotations = [[406,333], [0,0]]
        p_s = module.pseudo_labels(self, model, sample, weak_annotations)
        num_boxes = len(p_s[0][0])
        self.assertEqual(num_boxes, len(weak_annotations), "Number of bounding boxes should be equal to the number of weak annotations")

if __name__ == "__main__":
    unittest.main()