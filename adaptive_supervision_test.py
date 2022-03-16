import unittest
import math
from adaptive_supervision import dist_from_point

class TestDistFromPoint(unittest.TestCase):
    def test_dist_from_point(self):
        box = [0.0, 0.0, 4.0, 4.0]
        annotation = (3.0, 1.0)
        self.assertEqual(dist_from_point(box, annotation), math.sqrt(2), "Should be square root of 2")

if __name__ == "__main__":
    unittest.main()