#!/usr/bin/env python3
"""Run the TT-VLA synthetic reward smoke test or formal task matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from simpler_env.utils.synthetic_rewards import REWARD_ABLATION_TASKS


GPU_QUERY = (
    "index,uuid,memory.total,memory.used,utilization.gpu"
)


def query_gpus() -> list[dict[str, object]]:
    """Return current GPU capacity, memory use, and utilization."""
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes_by_uuid: dict[str, list[dict[str, object]]] = {}
    for line in process_result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, process_name, used_memory = [
            value.strip() for value in line.split(",", maxsplit=3)
        ]
        processes_by_uuid.setdefault(uuid, []).append(
            {
                "pid": int(pid),
                "process_name": process_name,
                "used_memory_mib": int(used_memory),
            }
        )

    gpus = []
    for line in result.stdout.splitlines():
        index, uuid, total, used, utilization = [
            value.strip() for value in line.split(",")
        ]
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "memory_free_mib": int(total) - int(used),
                "utilization_percent": int(utilization),
                "compute_processes": processes_by_uuid.get(uuid, []),
            }
        )
    return gpus


def select_gpu(
    min_free_memory_mib: int,
    gpu_index: int | None = None,
) -> tuple[int, list[dict[str, object]]]:
    """Select an available GPU, optionally requiring one physical index."""
    snapshot = query_gpus()
    candidates = [
        gpu
        for gpu in snapshot
        if gpu["memory_free_mib"] >= min_free_memory_mib
        and gpu["utilization_percent"] <= 5
        and not gpu["compute_processes"]
    ]
    if gpu_index is not None:
        requested = [gpu for gpu in snapshot if gpu["index"] == gpu_index]
        if not requested:
            raise ValueError(
                f"Requested physical GPU {gpu_index} was not found. Snapshot: {snapshot}"
            )
        if requested[0] not in candidates:
            raise RuntimeError(
                f"Requested physical GPU {gpu_index} does not currently satisfy "
                "the availability policy: "
                f"free_memory >= {min_free_memory_mib} MiB, utilization <= 5%, "
                f"and no compute process. GPU: {requested[0]}"
            )
        return gpu_index, snapshot
    if not candidates:
        raise RuntimeError(
            "No GPU currently satisfies the availability policy: "
            f"free_memory >= {min_free_memory_mib} MiB and utilization <= 5%. "
            f"Snapshot: {snapshot}"
        )
    selected = max(
        candidates,
        key=lambda gpu: (gpu["memory_free_mib"], -gpu["utilization_percent"]),
    )
    return int(selected["index"]), snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("synthetic_uniform", "synthetic_matched", "both"),
        default="synthetic_uniform",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-task-index", type=int, default=1)
    parser.add_argument("--vla-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-seed", type=int, default=0)
    parser.add_argument("--min-free-memory-mib", type=int, default=23_000)
    parser.add_argument(
        "--gpu-index",
        type=int,
        help="Require one physical GPU index instead of selecting any eligible GPU.",
    )
    return parser.parse_args()


def run_one(
    args: argparse.Namespace,
    mode: str,
    task_index: int,
) -> dict[str, object]:
    task = REWARD_ABLATION_TASKS[task_index]
    selected_gpu, before = select_gpu(args.min_free_memory_mib, args.gpu_index)
    episodes = 1 if args.smoke else 20
    task_slug = f"{task_index:02d}_{task.category}_{task.paper_name.lower().replace(' ', '_')}"
    run_dir = Path(args.output_root).expanduser().resolve() / mode / task_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        "simpler_env/train_ms3_ppo_ttt.py",
        "--name",
        f"{mode}-{task_slug}-seed{args.seed}",
        "--env_id",
        task.env_id,
        "--obj_set",
        task.obj_set,
        "--vla_path",
        args.vla_path,
        "--vla_unnorm_key",
        "bridge_orig",
        "--seed",
        str(args.seed),
        "--reward_seed",
        str(args.reward_seed),
        "--reward_mode",
        mode,
        "--reward_task_index",
        str(task_index),
        "--num_envs",
        "1",
        "--episode_len",
        "160",
        "--tt_steps",
        "8",
        "--max_episodes",
        str(episodes),
        "--vla_lora_rank",
        "32",
        "--buffer_minibatch",
        "8",
        "--output_dir",
        str(run_dir),
        "--no-wandb",
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(selected_gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        }
    )
    gpucomp = "/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.570.86.10"
    if Path(gpucomp).exists():
        env["LD_PRELOAD"] = gpucomp

    started = time.time()
    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    after = query_gpus()
    record = {
        "mode": mode,
        "task_index": task_index,
        "task": task.__dict__,
        "episodes": episodes,
        "requested_physical_gpu": args.gpu_index,
        "selected_physical_gpu": selected_gpu,
        "gpu_snapshot_before": before,
        "gpu_snapshot_after": after,
        "command": command,
        "output_dir": str(run_dir),
        "log_path": str(log_path),
        "exit_code": completed.returncode,
        "elapsed_seconds": time.time() - started,
    }
    with (run_dir / "launcher_result.json").open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Experiment failed for {mode} task {task_index}; see {log_path}"
        )
    return record


def main() -> None:
    args = parse_args()
    modes = (
        ("synthetic_uniform", "synthetic_matched")
        if args.mode == "both"
        else (args.mode,)
    )
    task_indices = (
        (args.smoke_task_index,)
        if args.smoke
        else tuple(range(len(REWARD_ABLATION_TASKS)))
    )
    records = []
    for mode in modes:
        for task_index in task_indices:
            records.append(run_one(args, mode, task_index))

    output_root = Path(args.output_root).expanduser().resolve()
    with (output_root / "experiment_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
