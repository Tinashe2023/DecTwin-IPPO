import psycopg2
import pandas as pd
import numpy as np

from env_config import get_db_params

DB_PARAMS = get_db_params()

if not DB_PARAMS.get("password"):
    raise RuntimeError(
        "Database password is not configured. Set DB_PASSWORD in the environment or .env file."
    )

EXPECTED_STEPS = 2000  # only sessions run under the fixed batch protocol are comparable


def load_data():
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql_query("SELECT * FROM factory_telemetry;", conn)
    conn.close()
    return df


def throughput_for_session(session_df):
    """PROCESS actions taken while not faulted, per step."""
    valid = session_df[(session_df["action_taken"] == 0) & (session_df["is_faulted"] == False)]
    max_steps = session_df["step_number"].max()
    return len(valid) / max_steps if max_steps > 0 else 0.0


def summarize(df, prefix, label):
    sessions = sorted(s for s in df["session_id"].unique() if s.startswith(prefix))
    values = []
    skipped = []
    print(f"\n=== {label} ({len(sessions)} session(s) found) ===")
    for sid in sessions:
        sdf = df[df["session_id"] == sid]
        actual_steps = sdf["step_number"].max()
        if actual_steps != EXPECTED_STEPS:
            skipped.append((sid, actual_steps))
            print(f"  {sid}: SKIPPED -- ran {actual_steps} steps, expected {EXPECTED_STEPS} (not comparable)")
            continue
        tp = throughput_for_session(sdf)
        values.append(tp)
        print(f"  {sid}: {tp:.3f}")

    if values:
        arr = np.array(values)
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        print(f"  --> mean = {arr.mean():.3f}, std = {std:.3f}, n = {len(arr)} (used) / {len(skipped)} (skipped)")
    else:
        print("  (no valid sessions found with this prefix)")
    return values


if __name__ == "__main__":
    print("[*] Connecting to database...")
    df = load_data()

    groups = [
        ("baseline_", "Rule-Based Baseline"),
        ("centralized_", "Centralized PPO (single policy, global state)"),
        ("noComm_", "Ablation: No Neighbor Observation (own state only)"),
        ("noPenalty_", "Ablation: No Standing Queue Penalty"),
        ("run_", "DecTwin-PPO (Decentralized IPPO, full model)"),
    ]

    results = {}
    for prefix, label in groups:
        results[label] = summarize(df, prefix, label)

    print("\n=== Summary (protocol-matched sessions only) ===")
    ref_label = "Rule-Based Baseline"
    ref_mean = np.mean(results[ref_label]) if results[ref_label] else None

    for label, values in results.items():
        if not values:
            continue
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        line = f"{label}: {mean:.3f} +/- {std:.3f} (n={len(values)})"
        if ref_mean and label != ref_label:
            pct = (mean - ref_mean) / ref_mean * 100
            line += f"  [{pct:+.1f}% vs. baseline]"
        print(line)
