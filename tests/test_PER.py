import unittest
import random

from util.prioritized_experience_replay import PrioritizedExperienceReplay
from util.sum_tree_buffer import SumTreeBuffer


class PERUnitTests(unittest.TestCase):
    def test_add(self):
        per = PrioritizedExperienceReplay(buffer_size=5)
        per.add(10, "a")
        v = per._get_priority(10)

        self.assertEqual("a", per._buffer._data[0])
        self.assertAlmostEqual(v, per._buffer._tree[0], places=4)
        self.assertAlmostEqual(v, per._buffer._tree[4], places=4)

    def test_get(self):
        per = PrioritizedExperienceReplay(buffer_size=5)
        per.add(10, "a")
        per.add(20, "b")
        v_a = per._get_priority(10)
        v_b = per._get_priority(20)
        total = per._buffer.total()
        N = 1000
        step = float(total) / N
        s = step
        print(per._buffer._data)
        print(per._buffer._tree)
        while s < v_a:
            self.assertEqual("a", per._buffer.get_entry(s)[1], msg="Out of bound access for value {0} out of {1}".format(s, total))
            s += step
        while s < v_b:
            self.assertEqual("b", per._buffer.get_entry(s)[1], msg="Out of bound access for value {0} out of {1}".format(s, total))
            s += step


if __name__ == '__main__':
    unittest.main()
