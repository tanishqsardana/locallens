from __future__ import annotations

import unittest

from videosearch.moment_overlay import active_moments_at_time


class MomentOverlayTest(unittest.TestCase):
    def test_active_moments_interval_and_point(self) -> None:
        moments = [
            {"type": "STOP", "start_time": 5.0, "end_time": 7.0, "entities": [1]},
            {"type": "APPEAR", "start_time": 6.0, "end_time": 6.0, "entities": [2]},
        ]
        at_55 = active_moments_at_time(moments, time_sec=5.5, point_tolerance_sec=0.0)
        self.assertEqual(len(at_55), 1)
        self.assertEqual(at_55[0]["type"], "STOP")

        at_60 = active_moments_at_time(moments, time_sec=6.0, point_tolerance_sec=0.0)
        self.assertEqual(len(at_60), 2)

        at_61 = active_moments_at_time(moments, time_sec=6.1, point_tolerance_sec=0.05)
        types = {row["type"] for row in at_61}
        self.assertIn("STOP", types)
        self.assertNotIn("APPEAR", types)


if __name__ == "__main__":
    unittest.main()
