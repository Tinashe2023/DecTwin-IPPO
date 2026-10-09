import os
import json
import asyncio
import websockets
import ray
import numpy as np
import psycopg2
import uuid

from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

from models.comm_net import CommNetGRU
from envs.factory_env import FactoryTwinEnv
from env_config import get_db_params

DB_PARAMS = get_db_params()

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except Exception as e:
        print(f"[-] Database connection failed: {e}")
        return None

def env_creator(config):
    env = FactoryTwinEnv(mode='inference')
    return ParallelPettingZooEnv(env)

async def run_digital_twin(websocket):
    print("\n[+] Web UI Connected! Booting up Digital Twin AI...")
    
    # Generate a unique ID for this specific run for the database
    session_id = f"run_{uuid.uuid4().hex[:8]}"
    db_conn = get_db_connection()
    if db_conn:
        print(f"[+] PostgreSQL Connected. Logging session: {session_id}")

    register_env("dectwin_v0", env_creator)
    ModelCatalog.register_custom_model("gru_comm_net", CommNetGRU)
    
    config = (
        PPOConfig()
        .environment(env="dectwin_v0")
        .framework("torch")
        .rollouts(num_rollout_workers=1)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        )
        .training(
            model={
                "custom_model": "gru_comm_net",
                "max_seq_len": 20,
            }
        )
    )
    
    algo = config.build()
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    
    try:
        algo.restore(checkpoint_dir)
        print("[+] AI Brain loaded successfully!")
    except Exception as e:
        print(f"[-] ERROR LOADING CHECKPOINT: {e}")
        return

    env = env_creator({})
    obs, info = env.reset()
    gru_states = {agent: [np.zeros(128, dtype=np.float32)] for agent in env.get_agent_ids()}
    step_count = 0
    
    try:
        while True:
            step_count += 1
            actions = {}
            
            for agent in env.get_agent_ids():
                action, state_out, _ = algo.compute_single_action(
                    observation=obs[agent],
                    state=gru_states[agent],
                    policy_id="shared_policy"
                )
                actions[agent] = action
                gru_states[agent] = state_out
            
            obs, rewards, terminations, truncations, infos = env.step(actions)
            base_env = env.par_env.unwrapped if hasattr(env.par_env, 'unwrapped') else env.par_env
            
            dashboard_data = {
                "step": step_count,
                "machines": []
            }
            
            # Prepare database batch insert
            db_records = []

            for agent in base_env.possible_agents:
                state = base_env.factory_state[agent]
                cpu_load = float(state[0])
                temperature = float(state[1])
                queue_depth = int(state[2])
                is_faulted = bool(state[3] == 1.0)
                last_action = int(actions.get(agent, -1))

                # Dashboard payload
                dashboard_data["machines"].append({
                    "id": agent,
                    "cpu_load": cpu_load,
                    "temperature": temperature,
                    "queue": queue_depth,
                    "is_faulted": is_faulted,
                    "last_action": last_action
                })

                # Database payload
                db_records.append((
                    session_id, step_count, agent, cpu_load, 
                    temperature, queue_depth, is_faulted, last_action
                ))
            
            # --- 2. EXECUTE DATABASE LOGGING ---
            if db_conn:
                try:
                    cursor = db_conn.cursor()
                    insert_query = """
                        INSERT INTO factory_telemetry 
                        (session_id, step_number, machine_id, cpu_load, temperature, queue_depth, is_faulted, action_taken)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.executemany(insert_query, db_records)
                    db_conn.commit()
                    cursor.close()
                except Exception as e:
                    print(f"[-] DB Write Error: {e}")

            # Broadcast to web UI
            await websocket.send(json.dumps(dashboard_data))
            await asyncio.sleep(0.5)
            
            if truncations[base_env.possible_agents[0]]:
                obs, info = env.reset()
                gru_states = {a: [np.zeros(128, dtype=np.float32)] for a in env.get_agent_ids()}
                
    except websockets.exceptions.ConnectionClosed:
        print("[-] Web UI Disconnected.")
    finally:
        if db_conn:
            db_conn.close()
            print("[-] Database connection closed.")
        algo.stop()

async def main():
    print("Initializing Ray...")
    ray.init(ignore_reinit_error=True, object_store_memory=200 * 1024 * 1024, _memory=2 * 1024 * 1024 * 1024)
    print("DIGITAL TWIN WEBSOCKET SERVER RUNNING ON PORT 8765")
    async with websockets.serve(run_digital_twin, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())