import os
import uuid
import argparse
import numpy as np
import psycopg2

import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

from env_config import get_db_params
from models.comm_net import CommNetGRU
from envs.factory_env import FactoryTwinEnv
from envs.no_comm_env import NoCommFactoryEnv

DB_PARAMS = get_db_params()

if not DB_PARAMS.get("password"):
    raise RuntimeError(
        "Database password is not configured. Set DB_PASSWORD in the environment or .env file."
    )

ABLATIONS = {
    "no_comm": {
        "env_class": NoCommFactoryEnv,
        "env_kwargs": {},
        "checkpoint_dir": "checkpoints_no_comm",
        "session_prefix": "noComm_",
    },
    "no_penalty": {
        "env_class": FactoryTwinEnv,
        "env_kwargs": {"queue_penalty_weight": 0.0},
        "checkpoint_dir": "checkpoints_no_penalty",
        "session_prefix": "noPenalty_",
    },
}

GRU_HIDDEN_SIZE = 128  # must match CommNetGRU.gru_hidden_size


def make_env_creator(spec):
    def env_creator(config):
        env = spec["env_class"](mode="inference", **spec["env_kwargs"])
        return ParallelPettingZooEnv(env)
    return env_creator


def run_ablation_eval(ablation: str, max_steps: int = 2000, deterministic: bool = True):
    spec = ABLATIONS[ablation]
    session_id = f"{spec['session_prefix']}{uuid.uuid4().hex[:8]}"

    print(f"\n[+] Starting '{ablation}' ablation batch evaluation...")
    try:
        db_conn = psycopg2.connect(**DB_PARAMS)
        cursor = db_conn.cursor()
        print(f"[+] PostgreSQL Connected. Logging session: {session_id}")
    except Exception as e:
        print(f"[-] Database connection failed: {e}")
        return

    ray.init(ignore_reinit_error=True, object_store_memory=200 * 1024 * 1024, _memory=2 * 1024 * 1024 * 1024)
    env_name = f"dectwin_{ablation}_v0"
    register_env(env_name, make_env_creator(spec))
    ModelCatalog.register_custom_model("gru_comm_net", CommNetGRU)

    config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        )
        .training(model={"custom_model": "gru_comm_net", "max_seq_len": 20})
    )

    algo = config.build()
    checkpoint_dir = os.path.join(os.getcwd(), spec["checkpoint_dir"])
    try:
        algo.restore(checkpoint_dir)
        print(f"[+] Loaded trained policy from: {checkpoint_dir}")
    except Exception as e:
        print(f"[-] ERROR LOADING CHECKPOINT: {e}")
        return

    raw_env = spec["env_class"](mode="inference", **spec["env_kwargs"])
    env = ParallelPettingZooEnv(raw_env)
    obs, info = env.reset()
    gru_states = {agent: [np.zeros(GRU_HIDDEN_SIZE, dtype=np.float32)] for agent in env.get_agent_ids()}

    step_count = 0
    mode_label = "deterministic (greedy)" if deterministic else "stochastic (sampled)"
    print(f"[*] Running {max_steps} steps [{mode_label}]...")

    while step_count < max_steps:
        step_count += 1
        actions = {}
        for agent in env.get_agent_ids():
            action, state_out, _ = algo.compute_single_action(
                observation=obs[agent],
                state=gru_states[agent],
                policy_id="shared_policy",
                explore=not deterministic,
            )
            actions[agent] = action
            gru_states[agent] = state_out

        obs, rewards, terminations, truncations, infos = env.step(actions)

        db_records = []
        for agent in raw_env.possible_agents:
            state = raw_env.factory_state[agent]
            db_records.append((
                session_id, step_count, agent,
                float(state[0]), float(state[1]), int(state[2]),
                bool(state[3] == 1.0), int(actions.get(agent, -1)),
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

        if truncations[raw_env.possible_agents[0]]:
            obs, info = env.reset()
            gru_states = {a: [np.zeros(GRU_HIDDEN_SIZE, dtype=np.float32)] for a in env.get_agent_ids()}

    cursor.close()
    db_conn.close()
    algo.stop()
    ray.shutdown()
    print(f"\n[+] '{ablation}' evaluation complete! {max_steps} steps logged under session '{session_id}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True, choices=list(ABLATIONS.keys()))
    args = parser.parse_args()
    run_ablation_eval(args.ablation)
