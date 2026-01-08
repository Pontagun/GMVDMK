import unittest

import numpy as np
import quaternion
import pipeline
import pipeline as pl

class PipelineTestCase(unittest.TestCase):

    def test_get_qref_w(self):
        x = [1, 0, 0]
        y = [0, 1, 0]

        self.assertEqual(pl.Correction.get_qref_w(x, y), 1)
