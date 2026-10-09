import psycopg2
import uuid
import numpy as np
from envs.factory_env import FactoryTwinEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from env_config import get_db_params

DB_PARAMS = get_db_params()

def run_baseline():
    print("\n[+] Starting Rule-Based Baseline Simulation...")
    
    # We label this session explicitly so we can filter it in SQL later
    session_id = f"baseline_{uuid.uuid4().hex[:8]}"
    
    try:
        db_conn = psycopg2.connect(**DB_PARAMS)
        cursor = db_conn.cursor()
        print(f"[+] PostgreSQL Connected. Logging baseline session: {session_id}")
    except Exception as e:
        print(f"[-] Database connection failed: {e}")
        return

    # Initialize the physical simulation
    raw_env = FactoryTwinEnv(mode='inference')
    env = ParallelPettingZooEnv(raw_env)
    
    obs, info = env.reset()
    step_count = 0
    max_baseline_steps = 2000 # Run for a solid chunk of time to get good statistical data
    
    print(f"[*] Running {max_baseline_steps} steps of rule-based routing...")
    
    while step_count < max_baseline_steps:
        step_count += 1
        actions = {}
        
        # RULE-BASED POLICY: 
        # If queue > 0, 70% chance to process, 30% chance to randomly reroute.
        # If no queue, sit idle.
        for agent in env.get_agent_ids():
            state = raw_env.factory_state[agent]
            queue_depth = state[2]
            is_faulted = (state[3] == 1.0)
            
            if is_faulted:
                actions[agent] = 1 # Idle while broken
            elif queue_depth > 0:
                actions[agent] = np.random.choice([0, 2, 3], p=[0.7, 0.15, 0.15])
            else:
                actions[agent] = 1 # Idle
                
        # Step the environment
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Prepare database batch insert
        db_records = []
        for agent in raw_env.possible_agents:
            state = raw_env.factory_state[agent]
            db_records.append((
                session_id, step_count, agent, float(state[0]), 
                float(state[1]), int(state[2]), bool(state[3] == 1.0), int(actions.get(agent, -1))
            ))
        
        # Log to Database
        try:
            insert_query = """
                INSERT INTO factory_telemetry 
                (session_id, step_number, machine_id, cpu_load, temperature, queue_depth, is_faulted, action_taken)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.executemany(insert_query, db_records)
            db_conn.commit()
        except Exception as e:
            print(f"[-] DB Write Error: {e}")
            
        if truncations[raw_env.possible_agents[0]]:
            obs, info = env.reset()

    if db_conn:
        cursor.close()
        db_conn.close()
        
    print(f"\n[+] Baseline Data Generation Complete! {max_baseline_steps} steps logged to PostgreSQL.")

if __name__ == "__main__":
    run_baseline()