import unittest
import numpy as np
from core.latent_reasoning import RULES, make_demo, run_recurrent_reasoner

class TestReasoning(unittest.TestCase):
    def test_identity_is_recovered(self):
        rule = next(r for r in RULES if r.name == "Identity")
        rng = np.random.default_rng(1)
        demos = [make_demo(rng, rule) for _ in range(3)]
        q, truth = make_demo(rng, rule)
        result = run_recurrent_reasoner(demos, q, iterations=8)
        self.assertEqual(result["rule"].name, "Identity")
        np.testing.assert_array_equal(result["prediction"], truth)

    def test_rotation_is_recovered(self):
        rule = next(r for r in RULES if r.name == "Rotate 90°")
        rng = np.random.default_rng(2)
        demos = [make_demo(rng, rule) for _ in range(3)]
        q, truth = make_demo(rng, rule)
        result = run_recurrent_reasoner(demos, q, iterations=8)
        self.assertEqual(result["rule"].name, "Rotate 90°")
        np.testing.assert_array_equal(result["prediction"], truth)

    def test_trajectory_shape(self):
        rule = next(r for r in RULES if r.name == "Flip vertical")
        rng = np.random.default_rng(3)
        demos = [make_demo(rng, rule) for _ in range(2)]
        q, _ = make_demo(rng, rule)
        result = run_recurrent_reasoner(demos, q, iterations=5)
        self.assertEqual(result["trajectory"].shape, (6, len(RULES)))

if __name__ == "__main__":
    unittest.main()
