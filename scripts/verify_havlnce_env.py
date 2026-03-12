#!/usr/bin/env python
"""Smoke-test HAVLNCEDaggerEnv over one or more episodes."""
import argparse
import os
import sys
import traceback

# 可选：降低 habitat-sim 的 glog 输出，避免刷屏（需在 import habitat 前设置）
if "GLOG_minloglevel" not in os.environ:
    os.environ["GLOG_minloglevel"] = "2"

# 确保从 agent 目录运行
AGENT_DIR = os.path.join(os.path.dirname(__file__), "..", "agent")
if os.getcwd() != os.path.abspath(AGENT_DIR):
    os.chdir(os.path.abspath(AGENT_DIR))
    print("CWD set to:", os.getcwd(), file=sys.stderr)

import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401

from vlnce_baselines.config.default import get_config
from vlnce_baselines.common.environments import HAVLNCEDaggerEnv


DEFAULT_ACTIONS = ["MOVE_FORWARD", "TURN_LEFT", "MOVE_FORWARD", "TURN_RIGHT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify HAVLNCEDaggerEnv across multiple episodes."
    )
    parser.add_argument(
        "--config-path",
        default="config/cma_pm_da_aug_tune.yaml",
        help="Path relative to HA-VLN/agent.",
    )
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to reset through.",
    )
    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=1,
        help="How many actions to execute after each reset.",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        default=DEFAULT_ACTIONS,
        help="Action cycle used for env.step(...).",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    return get_config(
        args.config_path,
        [
            "TASK_CONFIG.DATASET.SPLIT",
            args.split,
            "EVAL.SPLIT",
            args.split,
            "NUM_ENVIRONMENTS",
            "1",
        ],
    )


def summarize_metrics(metrics):
    summary = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, bool, str)):
            summary[key] = value
    return summary


def unpack_step_result(result):
    if isinstance(result, tuple) and len(result) == 4:
        return result
    return result, None, None, None


def main() -> int:
    args = parse_args()
    cfg = build_config(args)

    print("base_task:", cfg.BASE_TASK_CONFIG_PATH)
    print("split:", cfg.TASK_CONFIG.DATASET.SPLIT)
    print("human_glb_path:", cfg.TASK_CONFIG.SIMULATOR.HUMAN_GLB_PATH)
    print("human_counting:", cfg.TASK_CONFIG.SIMULATOR.HUMAN_COUNTING)
    print("episodes:", args.episodes)
    print("steps_per_episode:", args.steps_per_episode)
    print("actions:", args.actions)

    env = HAVLNCEDaggerEnv(cfg)
    failures = 0

    try:
        for episode_idx in range(args.episodes):
            try:
                obs = env.reset()
                current_episode = env._env.current_episode
                print(
                    f"[episode {episode_idx + 1}/{args.episodes}] "
                    f"reset_ok episode_id={current_episode.episode_id} "
                    f"scene={current_episode.scene_id}"
                )
                print("obs_keys:", sorted(obs.keys()))

                for step_idx in range(args.steps_per_episode):
                    action = args.actions[step_idx % len(args.actions)]
                    result = env.step(action)
                    obs, reward, done, info = unpack_step_result(result)
                    print(
                        f"[episode {episode_idx + 1}/{args.episodes}] "
                        f"step={step_idx + 1} action={action} "
                        f"done={done} reward={reward}"
                    )
                    if done:
                        break

                metrics = summarize_metrics(env.habitat_env.get_metrics())
                print(
                    f"[episode {episode_idx + 1}/{args.episodes}] "
                    f"metrics={metrics}"
                )
            except Exception:
                failures += 1
                print(
                    f"[episode {episode_idx + 1}/{args.episodes}] failed",
                    file=sys.stderr,
                )
                traceback.print_exc()
                if failures >= 1:
                    break
    finally:
        env.close()
        print("close_ok")

    print(
        f"summary: requested_episodes={args.episodes} "
        f"failures={failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
