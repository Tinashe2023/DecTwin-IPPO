import re
import pandas as pd
import psycopg2
import numpy as np

from env_config import get_db_params

DB_PARAMS = get_db_params()

if not DB_PARAMS.get("password"):
    raise RuntimeError(
        "Database password is not configured. Set DB_PASSWORD in the environment or .env file."
    )

EXPECTED_STEPS = 2000
SESSION_RE = re.compile(r"^sweepN(\d+)F(\d+)_")


def load_data():
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql_query("SELECT * FROM factory_telemetry;", conn)
    conn.close()
    return df


def per_machine_throughput(session_df, n_machines):
    """Normalized so different ring sizes are comparable: successful
    PROCESS actions per machine per step, not summed across all machines."""
    valid = session_df[(session_df["action_taken"] == 0) & (session_df["is_faulted"] == False)]
    max_steps = session_df["step_number"].max()
    return len(valid) / (max_steps * n_machines) if max_steps > 0 and n_machines > 0 else 0.0


def print_group(title, grouped):
    print(f"\n=== {title} ===")
    for key in sorted(grouped):
        vals = [v for _, v in grouped[key]]
        arr = np.array(vals)
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        print(f"  {key}: mean={arr.mean():.4f} +/- {std:.4f} (n={len(vals)})")
        for sid, v in grouped[key]:
            print(f"      {sid}: {v:.4f}")


if __name__ == "__main__":
    print("[*] Connecting to database...")
    df = load_data()

    scalability = {}       # n_machines -> [(session_id, per_machine_throughput)]
    fault_sensitivity = {}  # fault_pct -> [(session_id, per_machine_throughput)]

    sweep_sessions = [s for s in df["session_id"].unique() if SESSION_RE.match(s)]
    for sid in sorted(sweep_sessions):
        m = SESSION_RE.match(sid)
        n_machines, fault_pct = int(m.group(1)), int(m.group(2))
        sdf = df[df["session_id"] == sid]
        actual_steps = sdf["step_number"].max()
        if actual_steps != EXPECTED_STEPS:
            print(f"  {sid}: SKIPPED -- ran {actual_steps} steps, expected {EXPECTED_STEPS}")
            continue

        tp = per_machine_throughput(sdf, n_machines)
        if fault_pct == 100:  # default fault rate -> valid data point for the scalability axis
            scalability.setdefault(n_machines, []).append((sid, tp))
        if n_machines == 8:   # default ring size -> valid data point for the fault-rate axis
            fault_sensitivity.setdefault(fault_pct, []).append((sid, tp))

    # Fold in the existing full-model `run_*` sessions as the n=8 / 1.0x reference
    # point on both axes -- same model, same defaults, no need to re-run it.
    reference_sessions = [s for s in df["session_id"].unique() if s.startswith("run_")]
    for sid in sorted(reference_sessions):
        sdf = df[df["session_id"] == sid]
        if sdf["step_number"].max() != EXPECTED_STEPS:
            continue
        tp = per_machine_throughput(sdf, n_machines=8)
        scalability.setdefault(8, []).append((sid, tp))
        fault_sensitivity.setdefault(100, []).append((sid, tp))

    print_group("Scalability sweep (per-machine throughput, fault rate = 1.0x)", scalability)
    print_group("Fault-rate sensitivity (per-machine throughput, n_machines = 8)", fault_sensitivity)
