import unittest
import sys
import numpy as np

sys.path.append('../3Dresunet-Segmentation')

from model_py import UnetBuilder


class TestProject(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_numbers_3_4(self):
        self.assertEqual(np.multiply(3,4), 12)

if __name__ == '__main__':
    unittest.main()


