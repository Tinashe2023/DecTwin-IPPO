import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from env_config import get_db_params

DB_PARAMS = get_db_params()

def load_data():
    print("[*] Connecting to database to extract telemetry...")
    conn = psycopg2.connect(**DB_PARAMS)
    
    # Load all telemetry data into a Pandas DataFrame
    query = "SELECT * FROM factory_telemetry;"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def process_and_plot(df):
    print("[*] Processing metrics for publication figures...")
    
    # Identify the latest rule-based baseline and DecTwin-PPO (full model) sessions
    sessions = df['session_id'].unique()
    baseline_sessions = [s for s in sessions if s.startswith('baseline_')]
    marl_sessions = [s for s in sessions if s.startswith('run_')]
    
    if not baseline_sessions or not marl_sessions:
        print("[-] Error: Missing either baseline or MARL data in the database.")
        return

    latest_baseline = df[df['session_id'] == baseline_sessions[-1]].copy()
    latest_marl = df[df['session_id'] == marl_sessions[-1]].copy()

    # --- METRIC 1: SYSTEM THROUGHPUT ---
    # Throughput = (Total 'PROCESS' actions (0) taken while NOT faulted) / Max Steps
    def calc_throughput(data):
        valid_processes = data[(data['action_taken'] == 0) & (data['is_faulted'] == False)]
        max_steps = data['step_number'].max()
        return len(valid_processes) / max_steps if max_steps > 0 else 0

    tp_baseline = calc_throughput(latest_baseline)
    tp_marl = calc_throughput(latest_marl)

    def queue_stats(data, prefix):
        """Calculate mean and standard deviation of queue depth across sessions."""
        sessions = data[
            data['session_id'].astype(str).str.startswith(prefix)
        ]
        per_session = (
            sessions.groupby(['session_id', 'step_number'])['queue_depth']
            .mean()
            .reset_index()
        )
        pivot = per_session.pivot(
            index='step_number', columns='session_id', values='queue_depth'
        )
        return pivot.mean(axis=1), pivot.std(axis=1), pivot.shape[1]

    # Plot Figure 4: Queue Stress Across Sessions
    q_base_mean, q_base_std, n_base = queue_stats(df, 'baseline_')
    q_marl_mean, q_marl_std, n_marl = queue_stats(df, 'run_')

    plt.figure(figsize=(12, 5))
    plt.plot(
        q_base_mean.index, q_base_mean.values, color='#ef4444', alpha=0.9,
        label=f'Rule-Based Baseline (n={n_base})',
    )
    plt.fill_between(
        q_base_mean.index, q_base_mean - q_base_std,
        q_base_mean + q_base_std, color='#ef4444', alpha=0.15,
    )
    plt.plot(
        q_marl_mean.index, q_marl_mean.values, color='#0ea5e9', linewidth=2,
        label=f'DecTwin-PPO, Proposed (n={n_marl})',
    )
    plt.fill_between(
        q_marl_mean.index, q_marl_mean - q_marl_std,
        q_marl_mean + q_marl_std, color='#0ea5e9', alpha=0.15,
    )
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel('Average Factory Queue Depth (jobs)', fontsize=12)
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    plt.savefig('Figure_4_QueueStressAcrossSessions.png', dpi=300)
    print(
        f"[+] Saved Figure_4_QueueStressAcrossSessions.png "
        f"(baseline n={n_base}, marl n={n_marl})"
    )

    # --- METRIC 2: FAULT RECOVERY TIME (Approximation) ---
    # We approximate recovery by looking at queue depth reduction after a fault.
    def calc_queue_stress(data):
        # Calculate the average queue depth across all machines per step
        step_queues = data.groupby('step_number')['queue_depth'].mean()
        return step_queues

    q_baseline = calc_queue_stress(latest_baseline)
    q_marl = calc_queue_stress(latest_marl)

    # Plot Figure 3 Substitute: System Queue Stress Over Time
    plt.figure(figsize=(10, 5))
    plt.plot(q_baseline.index, q_baseline.values, label='Rule-Based Baseline', color='#ef4444', alpha=0.8)
    plt.plot(q_marl.index, q_marl.values, label='DecTwin-PPO (Proposed)', color='#0ea5e9', linewidth=2)
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel('Average Factory Queue Depth', fontsize=12)
    plt.ylim(0, 3.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Figure_3_QueueStress.png', dpi=300)
    print("[+] Saved Figure_3_QueueStress.png")

if __name__ == "__main__":
    try:
        df = load_data()
        process_and_plot(df)
        print("\n[+] Evaluation complete! Your paper figures are ready.")
    except Exception as e:
        print(f"[-] An error occurred: {e}")