#!/usr/bin/env python
import argparse
import base64
import io
import os
import sys
from copy import deepcopy
from typing import Any, Dict

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

if "GLOG_minloglevel" not in os.environ:
    os.environ["GLOG_minloglevel"] = "2"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGENT_DIR = os.path.join(ROOT_DIR, "agent")
VLNCE_DIR = os.path.join(AGENT_DIR, "VLN-CE")
for path in [VLNCE_DIR, ROOT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(AGENT_DIR)

import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401
from vlnce_baselines.common.environments import HAVLNCEDaggerEnv
from vlnce_baselines.config.default import get_config


ACTION_MAP = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}

app = Flask(__name__)
SERVER_STATE: Dict[str, Any] = {
    "env": None,
    "episodes": [],
    "episode_index": 0,
    "max_episodes": None,
    "last_obs": None,
    "last_metrics": {},
    "capabilities": {
        "has_external_lookdown_views": False,
    },
}


def current_havln_debug_state() -> Dict[str, Any]:
    env = SERVER_STATE.get("env")
    if env is None:
        return {}

    debug: Dict[str, Any] = {}
    havln_tool = getattr(env, "havlnce_tool", None)
    if havln_tool is not None:
        frame_id = getattr(havln_tool, "frame_id", None)
        total_signals_sent = getattr(havln_tool, "total_signals_sent", None)
        if frame_id is not None:
            debug["frame_id"] = int(frame_id)
        if total_signals_sent is not None:
            debug["total_signals_sent"] = int(total_signals_sent)

    sim = getattr(getattr(env, "_env", None), "_sim", None)
    human_positions = getattr(sim, "_human_positions", None) if sim is not None else None
    if isinstance(human_positions, dict):
        debug["human_count_live"] = len(human_positions)
        debug["human_positions"] = {
            viewpoint: {
                "position": np.asarray(position).tolist(),
                "rotation_euler": list(rotation_euler),
            }
            for viewpoint, (position, rotation_euler) in human_positions.items()
        }

    return debug


def encode_rgb(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_depth(depth: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, depth.astype(np.float32), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def summarize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    def to_jsonable(value: Any):
        if isinstance(value, dict):
            return {k: to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (int, float, bool, str)) or value is None:
            return value
        return str(value)

    summary = {key: to_jsonable(value) for key, value in metrics.items() if key != "top_down_map"}
    summary.setdefault("top_down_map", None)
    return summary


def current_episode_payload():
    episode = SERVER_STATE["env"].current_episode
    return {
        "episode_id": int(episode.episode_id),
        "scene_id": episode.scene_id,
        "instruction_text": episode.instruction.instruction_text,
    }


def obs_payload(obs, done: bool):
    metrics = summarize_metrics(SERVER_STATE["last_metrics"])
    debug_state = current_havln_debug_state()
    if debug_state:
        metrics["havln_debug_state"] = debug_state
    payload = current_episode_payload()
    payload.update(
        {
            "done": bool(done),
            "rgb_png_b64": encode_rgb(obs["rgb"]),
            "depth_npy_b64": encode_depth(obs["depth"]),
            "obs_keys": sorted(obs.keys()),
            "metrics": metrics,
            "capabilities": SERVER_STATE["capabilities"],
        }
    )
    if "lookdown_rgb" in obs:
        payload["lookdown_rgb_png_b64"] = encode_rgb(obs["lookdown_rgb"])
    if "lookdown_depth" in obs:
        payload["lookdown_depth_npy_b64"] = encode_depth(obs["lookdown_depth"])
    return payload


@app.route("/health", methods=["GET"])
def health():
    env = SERVER_STATE["env"]
    return jsonify(
        {
            "ok": True,
            "initialized": env is not None,
            "episode_index": SERVER_STATE["episode_index"],
            "total_episodes": len(SERVER_STATE["episodes"]),
            "max_episodes": SERVER_STATE["max_episodes"],
        }
    )


@app.route("/metadata", methods=["GET"])
def metadata():
    return jsonify(
        {
            "total_episodes": len(SERVER_STATE["episodes"]),
            "max_episodes": SERVER_STATE["max_episodes"],
            "capabilities": SERVER_STATE["capabilities"],
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    env = SERVER_STATE["env"]
    if env is None:
        return jsonify({"error": "env_not_initialized"}), 500

    episode_index = SERVER_STATE["episode_index"]
    max_episodes = SERVER_STATE["max_episodes"]
    if episode_index >= len(SERVER_STATE["episodes"]) or (max_episodes is not None and episode_index >= max_episodes):
        return jsonify(
            {
                "finished": True,
                "capabilities": SERVER_STATE["capabilities"],
                "episode_index": episode_index,
                "max_episodes": max_episodes,
            }
        )

    env._env.current_episode = SERVER_STATE["episodes"][episode_index]
    SERVER_STATE["episode_index"] += 1
    obs = env.reset()
    SERVER_STATE["last_obs"] = obs
    SERVER_STATE["last_metrics"] = env.habitat_env.get_metrics()
    payload = obs_payload(obs, done=False)
    payload["finished"] = False
    return jsonify(payload)


@app.route("/step", methods=["POST"])
def step():
    env = SERVER_STATE["env"]
    if env is None:
        return jsonify({"error": "env_not_initialized"}), 500

    action_code = int(request.json["action"])
    if action_code not in ACTION_MAP:
        return jsonify({"error": f"unsupported_action:{action_code}"}), 400

    obs, reward, done, info = env.step(ACTION_MAP[action_code])
    SERVER_STATE["last_obs"] = obs
    SERVER_STATE["last_metrics"] = info
    payload = obs_payload(obs, done=done)
    payload["reward"] = float(reward)
    return jsonify(payload)


@app.route("/close", methods=["POST"])
def close():
    env = SERVER_STATE["env"]
    if env is not None:
        env.close()
        SERVER_STATE["env"] = None
    return jsonify({"closed": True})


def build_env(args):
    cfg = get_config(
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
    if args.enable_lookdown_sensors:
        cfg.defrost()
        add_fixed_lookdown_sensors_to_config(cfg, tilt_degrees=args.lookdown_degrees)
        cfg.freeze()
    env = HAVLNCEDaggerEnv(cfg)
    SERVER_STATE["env"] = env
    SERVER_STATE["episodes"] = list(env.episodes)
    SERVER_STATE["episode_index"] = 0
    SERVER_STATE["max_episodes"] = args.max_episodes
    SERVER_STATE["capabilities"] = {
        "has_external_lookdown_views": bool(args.enable_lookdown_sensors),
        "lookdown_degrees": float(args.lookdown_degrees) if args.enable_lookdown_sensors else None,
        "split": args.split,
    }


def add_fixed_lookdown_sensors_to_config(cfg, tilt_degrees: float) -> None:
    pitch_radians = -np.deg2rad(float(tilt_degrees))

    if "RGB_SENSOR" in cfg.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS:
        lookdown_rgb = deepcopy(cfg.TASK_CONFIG.SIMULATOR.RGB_SENSOR)
        lookdown_rgb.UUID = "lookdown_rgb"
        lookdown_rgb.ORIENTATION = (pitch_radians, 0.0, 0.0)
        cfg.TASK_CONFIG.SIMULATOR.RGB_LOOKDOWN_SENSOR = lookdown_rgb
        cfg.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append("RGB_LOOKDOWN_SENSOR")

    if "DEPTH_SENSOR" in cfg.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS:
        lookdown_depth = deepcopy(cfg.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR)
        lookdown_depth.UUID = "lookdown_depth"
        lookdown_depth.ORIENTATION = (pitch_radians, 0.0, 0.0)
        cfg.TASK_CONFIG.SIMULATOR.DEPTH_LOOKDOWN_SENSOR = lookdown_depth
        cfg.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append("DEPTH_LOOKDOWN_SENSOR")

    cfg.SENSORS = cfg.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS


def parse_args():
    parser = argparse.ArgumentParser(description="Serve HA-VLN HAVLNCEDaggerEnv over HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--config-path", default="config/cma_pm_da_aug_tune.yaml")
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--enable-lookdown-sensors", action="store_true", default=True)
    parser.add_argument("--disable-lookdown-sensors", action="store_false", dest="enable_lookdown_sensors")
    parser.add_argument("--lookdown-degrees", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    build_env(args)
    # Habitat-Sim/OpenGL context is thread-affine. Keep request handling in the
    # main thread to avoid `GL::Context::current(): no current context`.
    app.run(
        host=args.host,
        port=args.port,
        threaded=False,
        processes=1,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
