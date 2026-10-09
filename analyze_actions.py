import pandas as pd
import psycopg2

from env_config import get_db_params

DB_PARAMS = get_db_params()

if not DB_PARAMS.get("password"):
    raise RuntimeError(
        "Database password is not configured. Set DB_PASSWORD in the environment or .env file."
    )

EXPECTED_STEPS = 2000
GROUPS = [
    ("baseline_", "Rule-Based Baseline"),
    ("centralized_", "Centralized PPO"),
    ("noComm_", "Ablation: No Neighbor Observation"),
    ("noPenalty_", "Ablation: No Standing Queue Penalty"),
    ("run_", "DecTwin-PPO (Full Model)"),
]
ACTION_NAMES = {0: "PROCESS", 1: "IDLE", 2: "REROUTE_LEFT", 3: "REROUTE_RIGHT"}


def load_data():
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql_query("SELECT * FROM factory_telemetry;", conn)
    conn.close()
    return df


def analyze_group(df, prefix, label):
    sessions = [s for s in df["session_id"].unique() if s.startswith(prefix)]
    clean_sessions = [
        s for s in sessions
        if df[df["session_id"] == s]["step_number"].max() == EXPECTED_STEPS
    ]
    if not clean_sessions:
        return

    sub = df[df["session_id"].isin(clean_sessions)]
    print(f"\n=== {label} (n={len(clean_sessions)} clean sessions, {len(sub)} machine-steps) ===")

    action_frac = sub["action_taken"].value_counts(normalize=True).sort_index()
    for a, frac in action_frac.items():
        name = ACTION_NAMES.get(a, f"action_{a}")
        print(f"  {name}: {frac * 100:.1f}%")

    print(f"  Avg queue depth (per machine): {sub['queue_depth'].mean():.2f}")
    print(f"  Fraction of machine-steps faulted: {(sub['is_faulted'] == True).mean() * 100:.1f}%")

    process_rows = sub[sub["action_taken"] == 0]
    if len(process_rows):
        wasted_proxy = (process_rows["queue_depth"] == 0).mean()
        print(f"  Of PROCESS actions, fraction resulting in queue_depth==0 afterward: {wasted_proxy * 100:.1f}%")
        print("    (high + a near-100% PROCESS rate above = suggests spamming PROCESS")
        print("     regardless of whether there was a job to clear)")


if __name__ == "__main__":
    print("[*] Connecting to database...")
    df = load_data()
    for prefix, label in GROUPS:
        analyze_group(df, prefix, label)
