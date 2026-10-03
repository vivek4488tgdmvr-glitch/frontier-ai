import unittest
import numpy as np
from core.bdh_bridge import hebbian_write, read_synapses

class TestBDHBridge(unittest.TestCase):
    def test_outer_product_write(self):
        sigma = np.zeros((3, 3))
        x = np.array([1.0, 0.0, 1.0])
        v = np.array([0.0, 1.0, 0.0])
        out = hebbian_write(sigma, x, v, decay=0.0)
        expected = np.outer(x, v)
        np.testing.assert_allclose(out, expected)

    def test_read(self):
        sigma = np.eye(3)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(read_synapses(x, sigma), x)

if __name__ == "__main__":
    unittest.main()
