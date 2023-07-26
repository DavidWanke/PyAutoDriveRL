import unittest
from src.env.model.simulation.utils import get_merged_threshold_intervals
from src.env.model.simulation.utils import get_merged_overlapping_intervals
from src.env.model.simulation.utils import get_velocity_time_way_acceleration


class TestIntervals(unittest.TestCase):

    def test_merge_overlapping_intervals(self):
        unsorted_intervals = [[1, 2], [-1, 1], [4, 6], [3, 9], [10, 12]]
        solution = [[-1, 2], [3, 9], [10, 12]]

        self.assertEqual(solution, get_merged_overlapping_intervals(unsorted_intervals))

    def test_merge_threshold_intervals(self):
        threshold = 1
        merged_intervals = [[-1, 2], [3, 9], [11, 13]]
        solution = [[-1, 9], [11, 13]]

        self.assertEqual(solution, get_merged_threshold_intervals(merged_intervals, threshold=threshold))

        threshold = 2
        merged_intervals = [[-1, 2], [3, 9], [11, 13]]
        solution = [[-1, 13]]

        self.assertEqual(solution, get_merged_threshold_intervals(merged_intervals, threshold=threshold))

    def test_get_velocity_time_way_acceleration(self):
        s = get_velocity_time_way_acceleration(s=None, a=23, t=1, v_0=50)

        self.assertEqual(s, 61.5)

        a = get_velocity_time_way_acceleration(s=23, a=None, t=2, v_0=10)

        self.assertEqual(a, 1.5)

        t = get_velocity_time_way_acceleration(s=40, a=3, t=None, v_0=10)

        self.assertEqual(round(t[0], 3), round(2.8130296381953, 3))
        self.assertEqual(round(t[1], 3), round(-9.4796963, 3))

        v = get_velocity_time_way_acceleration(s=20, a=3, t=3, v_0=None)

        self.assertEqual(round(v, 3), round(2.166666, 3))


if __name__ == '__main__':
    unittest.main()
