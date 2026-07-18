"""Tests for the fixed TT-VLA synthetic reward schedules."""

import unittest

import numpy as np

from simpler_env.utils.synthetic_rewards import (
    NEGATIVE_COUNT,
    NEGATIVE_MAX_MAGNITUDE,
    NEGATIVE_MEAN_MAGNITUDE,
    NEGATIVE_MEDIAN_MAGNITUDE,
    NEGATIVE_P95_MAGNITUDE,
    POSITIVE_COUNT,
    POSITIVE_MAX_MAGNITUDE,
    POSITIVE_MEAN_MAGNITUDE,
    POSITIVE_MEDIAN_MAGNITUDE,
    POSITIVE_P95_MAGNITUDE,
    STEPS_PER_TASK,
    TASK_COUNT,
    TOTAL_REWARD_COUNT,
    ZERO_COUNT,
    build_reward_schedule,
    build_task_reward_schedule,
    reward_schedule_sha256,
)


class SyntheticRewardScheduleTest(unittest.TestCase):
    def test_exact_class_counts_and_shared_positions(self):
        matched = build_reward_schedule("synthetic_matched", seed=7)
        uniform = build_reward_schedule("synthetic_uniform", seed=7)

        self.assertEqual(matched.size, TOTAL_REWARD_COUNT)
        self.assertEqual(np.count_nonzero(matched > 0), POSITIVE_COUNT)
        self.assertEqual(np.count_nonzero(matched < 0), NEGATIVE_COUNT)
        self.assertEqual(np.count_nonzero(matched == 0), ZERO_COUNT)
        np.testing.assert_array_equal(np.sign(matched), np.sign(uniform))

    def test_matched_statistics(self):
        rewards = build_reward_schedule("synthetic_matched", seed=11)
        positive = rewards[rewards > 0].astype(np.float64)
        negative = np.abs(rewards[rewards < 0].astype(np.float64))

        for values, targets in (
            (
                positive,
                (
                    POSITIVE_MEAN_MAGNITUDE,
                    POSITIVE_MEDIAN_MAGNITUDE,
                    POSITIVE_P95_MAGNITUDE,
                    POSITIVE_MAX_MAGNITUDE,
                ),
            ),
            (
                negative,
                (
                    NEGATIVE_MEAN_MAGNITUDE,
                    NEGATIVE_MEDIAN_MAGNITUDE,
                    NEGATIVE_P95_MAGNITUDE,
                    NEGATIVE_MAX_MAGNITUDE,
                ),
            ),
        ):
            actual = (
                values.mean(),
                np.median(values),
                np.quantile(values, 0.95),
                values.max(),
            )
            np.testing.assert_allclose(actual, targets, rtol=0, atol=1e-6)

    def test_uniform_ranges_and_reproducibility(self):
        first = build_reward_schedule("synthetic_uniform", seed=23)
        second = build_reward_schedule("synthetic_uniform", seed=23)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(reward_schedule_sha256(first), reward_schedule_sha256(second))

        positive = first[first > 0]
        negative = first[first < 0]
        self.assertTrue(np.all(positive > 0))
        self.assertTrue(np.all(positive <= POSITIVE_MAX_MAGNITUDE))
        self.assertTrue(np.all(negative < 0))
        self.assertTrue(np.all(negative >= -NEGATIVE_MAX_MAGNITUDE))

    def test_task_slices_cover_schedule_without_overlap(self):
        full = build_reward_schedule("synthetic_uniform", seed=31)
        slices = [
            build_task_reward_schedule("synthetic_uniform", seed=31, task_index=index)
            for index in range(TASK_COUNT)
        ]
        self.assertTrue(all(values.size == STEPS_PER_TASK for values in slices))
        np.testing.assert_array_equal(np.concatenate(slices), full)

    def test_invalid_task_index_is_rejected(self):
        with self.assertRaises(ValueError):
            build_task_reward_schedule("synthetic_uniform", seed=0, task_index=-1)
        with self.assertRaises(ValueError):
            build_task_reward_schedule(
                "synthetic_uniform",
                seed=0,
                task_index=TASK_COUNT,
            )


if __name__ == "__main__":
    unittest.main()
