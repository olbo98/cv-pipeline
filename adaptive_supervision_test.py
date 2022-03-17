from __future__ import annotations
import unittest
import math
from adaptive_supervision import dist_from_point, find_closest_box

class TestDistFromPoint(unittest.TestCase):
    def test_dist_from_point(self):
        box = [0.0, 0.0, 4.0, 4.0]
        annotation = (3.0, 1.0)
        self.assertEqual(dist_from_point(box, annotation), math.sqrt(2), "Should be square root of 2")

class TestFindClosestBox(unittest.TestCase):
    def test_find_closes_box(self):
        boxes = [[0.0, 0.0, 4.0, 4.0], [6.0, 5.0, 8.0, 7.0], [3.0, 1.0, 7.0, 5.0]]
        annotation = (4.0, 4.0)
        self.assertEqual(find_closest_box(boxes, annotation), boxes[2], "Should be [3.0, 1.0, 7.0, 5.0]")

if __name__ == "__main__":
    unittest.main()