"""Synthetic reward schedules for the TT-VLA reward ablation.

The formal experiment contains 15 tasks, 20 episodes per task, and 160
environment steps per episode.  This produces exactly 48,000 rewards per
experimental group, matching the sample counts supplied for the ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

import numpy as np


SyntheticRewardMode = Literal["synthetic_matched", "synthetic_uniform"]

TASK_COUNT = 15
EPISODES_PER_TASK = 20
STEPS_PER_EPISODE = 160
STEPS_PER_TASK = EPISODES_PER_TASK * STEPS_PER_EPISODE
TOTAL_REWARD_COUNT = TASK_COUNT * STEPS_PER_TASK

POSITIVE_COUNT = 5_476
NEGATIVE_COUNT = 1_590
ZERO_COUNT = 40_934

POSITIVE_MEAN_MAGNITUDE = 0.038344
POSITIVE_MEDIAN_MAGNITUDE = 0.019016
POSITIVE_P95_MAGNITUDE = 0.160015
POSITIVE_MAX_MAGNITUDE = 0.885473

NEGATIVE_MEAN_MAGNITUDE = 0.032418
NEGATIVE_MEDIAN_MAGNITUDE = 0.016258
NEGATIVE_P95_MAGNITUDE = 0.149905
NEGATIVE_MAX_MAGNITUDE = 0.332482


@dataclass(frozen=True)
class RewardAblationTask:
    """One paper task, identified by its environment and object set."""

    category: str
    paper_name: str
    env_id: str
    obj_set: str


REWARD_ABLATION_TASKS = (
    RewardAblationTask("execution", "Obj. Pos.", "PutOnPlateInScene25Position-v1", "test"),
    RewardAblationTask("execution", "Robot Pose", "PutOnPlateInScene25EEPose-v1", "test"),
    RewardAblationTask(
        "execution",
        "Obj. Rep.",
        "PutOnPlateInScene25PositionChangeTo-v1",
        "test",
    ),
    RewardAblationTask("vision", "Table", "PutOnPlateInScene25VisionImage-v1", "test"),
    RewardAblationTask(
        "vision", "Texture-w", "PutOnPlateInScene25VisionTexture03-v1", "test"
    ),
    RewardAblationTask("vision", "Noise-w", "PutOnPlateInScene25VisionWhole03-v1", "test"),
    RewardAblationTask(
        "vision", "Texture-s", "PutOnPlateInScene25VisionTexture05-v1", "test"
    ),
    RewardAblationTask("vision", "Noise-s", "PutOnPlateInScene25VisionWhole05-v1", "test"),
    RewardAblationTask(
        "semantics", "M-Obj. OOD", "PutOnPlateInScene25MultiCarrot-v1", "test"
    ),
    RewardAblationTask("semantics", "Instruct", "PutOnPlateInScene25Instruct-v1", "test"),
    RewardAblationTask(
        "semantics", "M Recep.", "PutOnPlateInScene25MultiPlate-v1", "test"
    ),
    RewardAblationTask("semantics", "Recep.", "PutOnPlateInScene25Plate-v1", "test"),
    RewardAblationTask(
        "semantics", "Dist Recep.", "PutOnPlateInScene25MultiPlate-v1", "train"
    ),
    RewardAblationTask("semantics", "Object", "PutOnPlateInScene25Carrot-v1", "test"),
    RewardAblationTask(
        "semantics", "M-Obj. IND", "PutOnPlateInScene25MultiCarrot-v1", "train"
    ),
)


def _build_magnitude_pool(
    count: int,
    mean: float,
    median: float,
    p95: float,
    maximum: float,
) -> np.ndarray:
    """Build a deterministic four-level pool matching requested statistics.

    The pool is intentionally synthetic: it pins the sample median, linear
    sample P95, and maximum, then solves the middle plateau analytically so the
    sample mean also matches.  It does not claim to reconstruct original data.
    """
    if count <= 2 or count % 2:
        raise ValueError("Magnitude pool count must be an even integer greater than 2")
    if not 0 < median <= p95 <= maximum:
        raise ValueError("Expected 0 < median <= p95 <= maximum")

    median_high_index = count // 2
    p95_low_index = int(np.floor((count - 1) * 0.95))

    lower_count = median_high_index + 1
    middle_count = p95_low_index - lower_count
    upper_count = count - 1 - p95_low_index
    if middle_count <= 0:
        raise ValueError("Magnitude pool is too small to define the requested quantiles")

    middle = (
        count * mean
        - lower_count * median
        - upper_count * p95
        - maximum
    ) / middle_count
    if not median <= middle <= p95:
        raise ValueError(
            "Requested statistics cannot be represented by the deterministic pool: "
            f"middle={middle}"
        )

    pool = np.empty(count, dtype=np.float64)
    pool[:lower_count] = median
    pool[lower_count:p95_low_index] = middle
    pool[p95_low_index:-1] = p95
    pool[-1] = maximum
    return pool


def build_sign_schedule(seed: int) -> np.ndarray:
    """Return the shared shuffled sign schedule with exact class counts."""
    signs = np.concatenate(
        (
            np.ones(POSITIVE_COUNT, dtype=np.int8),
            -np.ones(NEGATIVE_COUNT, dtype=np.int8),
            np.zeros(ZERO_COUNT, dtype=np.int8),
        )
    )
    if signs.size != TOTAL_REWARD_COUNT:
        raise AssertionError("Reward class counts do not sum to 48,000")
    np.random.default_rng(np.random.SeedSequence([seed, 0])).shuffle(signs)
    return signs


def build_reward_schedule(mode: SyntheticRewardMode, seed: int) -> np.ndarray:
    """Build one complete 48,000-step synthetic reward schedule."""
    if mode not in {"synthetic_matched", "synthetic_uniform"}:
        raise ValueError(f"Unsupported synthetic reward mode: {mode}")

    signs = build_sign_schedule(seed)
    rewards = np.zeros(TOTAL_REWARD_COUNT, dtype=np.float64)
    positive_mask = signs > 0
    negative_mask = signs < 0
    magnitude_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))

    if mode == "synthetic_uniform":
        # 1-U maps NumPy's [0, 1) samples to (0, 1], including the requested
        # maximum while excluding an exact zero magnitude.
        positive_magnitudes = POSITIVE_MAX_MAGNITUDE * (
            1.0 - magnitude_rng.random(POSITIVE_COUNT)
        )
        negative_magnitudes = NEGATIVE_MAX_MAGNITUDE * (
            1.0 - magnitude_rng.random(NEGATIVE_COUNT)
        )
    else:
        positive_magnitudes = _build_magnitude_pool(
            POSITIVE_COUNT,
            POSITIVE_MEAN_MAGNITUDE,
            POSITIVE_MEDIAN_MAGNITUDE,
            POSITIVE_P95_MAGNITUDE,
            POSITIVE_MAX_MAGNITUDE,
        )
        negative_magnitudes = _build_magnitude_pool(
            NEGATIVE_COUNT,
            NEGATIVE_MEAN_MAGNITUDE,
            NEGATIVE_MEDIAN_MAGNITUDE,
            NEGATIVE_P95_MAGNITUDE,
            NEGATIVE_MAX_MAGNITUDE,
        )
        magnitude_rng.shuffle(positive_magnitudes)
        magnitude_rng.shuffle(negative_magnitudes)

    rewards[positive_mask] = positive_magnitudes
    rewards[negative_mask] = -negative_magnitudes
    return rewards.astype(np.float32)


def build_task_reward_schedule(
    mode: SyntheticRewardMode,
    seed: int,
    task_index: int,
) -> np.ndarray:
    """Return the 3,200 rewards assigned to one task in the fixed manifest."""
    if not 0 <= task_index < TASK_COUNT:
        raise ValueError(f"reward_task_index must be in [0, {TASK_COUNT - 1}]")
    start = task_index * STEPS_PER_TASK
    stop = start + STEPS_PER_TASK
    return build_reward_schedule(mode, seed)[start:stop].copy()


def validate_task_selection(task_index: int, env_id: str, obj_set: str) -> None:
    """Prevent an experiment shard from being assigned to the wrong task."""
    if not 0 <= task_index < TASK_COUNT:
        raise ValueError(f"reward_task_index must be in [0, {TASK_COUNT - 1}]")
    task = REWARD_ABLATION_TASKS[task_index]
    if (env_id, obj_set) != (task.env_id, task.obj_set):
        raise ValueError(
            f"Task index {task_index} is {task.env_id}/{task.obj_set}, got "
            f"{env_id}/{obj_set}"
        )


def summarize_rewards(rewards: np.ndarray) -> dict[str, object]:
    """Return class counts and conditional magnitude statistics."""
    values = np.asarray(rewards, dtype=np.float64).reshape(-1)
    result: dict[str, object] = {
        "count": int(values.size),
        "mean": float(values.mean()) if values.size else 0.0,
        "min": float(values.min()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
    }
    for name, mask in (
        ("positive", values > 0),
        ("negative", values < 0),
        ("zero", values == 0),
    ):
        selected = np.abs(values[mask])
        stats: dict[str, float | int] = {"count": int(selected.size)}
        if selected.size:
            stats.update(
                {
                    "conditional_mean_magnitude": float(selected.mean()),
                    "median_magnitude": float(np.median(selected)),
                    "p95_magnitude": float(np.quantile(selected, 0.95)),
                    "max_magnitude": float(selected.max()),
                }
            )
        result[name] = stats
    return result


def reward_schedule_sha256(rewards: np.ndarray) -> str:
    """Return a stable digest for an already materialized reward schedule."""
    values = np.asarray(rewards, dtype=np.float32).reshape(-1)
    return hashlib.sha256(values.tobytes()).hexdigest()
