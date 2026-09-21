"""v1 shared-prefix mask tests — pure logic, no weights needed."""
import unittest


class TestBlockMask(unittest.TestCase):
    def test_single_branch_is_pure_causal(self):
        from smalljev.parallel import build_block_mask
        import numpy as np
        m = build_block_mask(3, [4])
        causal = np.tri(7, 7, dtype=bool)
        self.assertTrue(bool((m == causal).all()))

    def test_branches_cannot_see_each_other(self):
        from smalljev.parallel import build_block_mask
        # state=2, B1=[2,4), B2=[4,6)
        m = build_block_mask(2, [2, 2])
        # B2 query sees state + own past, not B1
        self.assertTrue(m[4, 0] and m[4, 1])   # state visible
        self.assertFalse(m[4, 2])              # B1 hidden from B2
        self.assertFalse(m[4, 3])
        self.assertTrue(m[4, 4])               # self visible
        self.assertFalse(m[5, 2])              # B1 hidden incl. from later B2 token
        self.assertTrue(m[5, 4])               # own past visible
        # B1 query sees state + own past, not B2 (incl. future)
        self.assertTrue(m[2, 0] and m[3, 2])
        self.assertFalse(m[2, 4])
        self.assertFalse(m[0, 2])              # state cannot see future branches

    def test_state_is_causal_within_itself(self):
        from smalljev.parallel import build_block_mask
        m = build_block_mask(3, [2])
        self.assertFalse(m[0, 1])
        self.assertTrue(m[1, 0])

    def test_readout_positions_are_branch_ends(self):
        from smalljev.parallel import readouts
        self.assertEqual(readouts(5, [3, 4, 2]), [7, 11, 13])

    def test_additive_mask_dtype_and_shape(self):
        from smalljev.parallel import build_block_mask, to_additive_4d
        import numpy as np
        m = build_block_mask(4, [3, 3])
        add = to_additive_4d(m)
        self.assertEqual(add.shape, (1, 1, 10, 10))
        self.assertEqual(float(add[0, 0, 5, 0]), 0.0)   # B1 sees state
        self.assertEqual(float(add[0, 0, 5, 4]), 0.0)   # B1 sees own past
        self.assertEqual(float(add[0, 0, 7, 0]), 0.0)   # B2 sees state
        self.assertTrue(float(add[0, 0, 7, 4]) < -1e4)  # B2 cannot see B1


if __name__ == "__main__":
    unittest.main()
