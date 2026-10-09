import os
import uuid
import numpy as np
import psycopg2

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from envs.centralized_env import CentralizedFactoryEnv
from env_config import get_db_params

DB_PARAMS = get_db_params()


def env_creator(config):
    return CentralizedFactoryEnv(mode="inference")


def run_centralized_eval(max_steps: int = 2000, checkpoint_dir: str = None, deterministic: bool = True):
    print("\n[+] Starting Centralized-PPO Batch Evaluation...")
    session_id = f"centralized_{uuid.uuid4().hex[:8]}"

    try:
        db_conn = psycopg2.connect(**DB_PARAMS)
        cursor = db_conn.cursor()
        print(f"[+] PostgreSQL Connected. Logging centralized session: {session_id}")
    except Exception as e:
        print(f"[-] Database connection failed: {e}")
        return

    ray.init(ignore_reinit_error=True, object_store_memory=200 * 1024 * 1024, _memory=2 * 1024 * 1024 * 1024)
    env_name = "dectwin_centralized_v0"
    register_env(env_name, env_creator)

    config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .training(model={"fcnet_hiddens": [256, 256], "fcnet_activation": "tanh"})
    )

    algo = config.build()
    checkpoint_dir = checkpoint_dir or os.path.join(os.getcwd(), "checkpoints_centralized")
    try:
        algo.restore(checkpoint_dir)
        print(f"[+] Loaded trained centralized policy from: {checkpoint_dir}")
    except Exception as e:
        print(f"[-] ERROR LOADING CHECKPOINT: {e}")
        return

    env = env_creator({})
    obs, info = env.reset()
    possible_agents = env._env.possible_agents

    step_count = 0
    mode_label = "deterministic (greedy)" if deterministic else "stochastic (sampled)"
    print(f"[*] Running {max_steps} steps of the centralized policy [{mode_label}]...")

    while step_count < max_steps:
        step_count += 1
        action = algo.compute_single_action(observation=obs, explore=not deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        db_records = []
        for i, agent in enumerate(possible_agents):
            state = env._env.factory_state[agent]
            db_records.append((
                session_id, step_count, agent,
                float(state[0]), float(state[1]), int(state[2]),
                bool(state[3] == 1.0), int(action[i]),
            ))

        try:
            cursor.executemany(
                """
                INSERT INTO factory_telemetry
                (session_id, step_number, machine_id, cpu_load, temperature, queue_depth, is_faulted, action_taken)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                db_records,
            )
            db_conn.commit()
        except Exception as e:
            print(f"[-] DB Write Error: {e}")

        if truncated or terminated:
            obs, info = env.reset()

    cursor.close()
    db_conn.close()
    algo.stop()
    ray.shutdown()
    print(f"\n[+] Centralized Evaluation Complete! {max_steps} steps logged under session '{session_id}'.")


if __name__ == "__main__":
    run_centralized_eval()
