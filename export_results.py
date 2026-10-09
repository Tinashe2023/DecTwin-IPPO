"""
Export per-session evaluation results from PostgreSQL to CSV so results can be
inspected without running the database.

    python export_results.py            # writes results/results_per_session.csv
    python export_results.py --full     # also writes results/telemetry_matched.csv.gz
                                        # (full per-step telemetry for protocol-matched sessions;
                                        #  large -- intended for a data repository, not git)

A session is "protocol-matched" if it ran exactly EXPECTED_STEPS steps. Other
sessions (e.g. earlier live-demo runs) are listed with protocol_matched=False
and must not be used in reported statistics.
"""
import os
import re
import argparse
import pandas as pd
import psycopg2

from env_config import get_db_params

DB_PARAMS = get_db_params()
if not DB_PARAMS.get("password"):
    raise RuntimeError(
        "Database password is not configured. Set DB_PASSWORD in the environment or .env file."
    )

EXPECTED_STEPS = 2000
ACTION_NAMES = {0: "PROCESS", 1: "IDLE", 2: "REROUTE_LEFT", 3: "REROUTE_RIGHT"}
PREFIX_CONDITIONS = [
    ("baseline_", "rule_based_baseline"),
    ("centralized_", "centralized_ppo"),
    ("noComm_", "ablation_no_neighbor_observation"),
    ("noPenalty_", "ablation_no_queue_penalty"),
    ("run_", "full_model"),
]
SWEEP_RE = re.compile(r"^sweepN(\d+)F(\d+)_")


def classify(session_id):
    m = SWEEP_RE.match(session_id)
    if m:
        return "full_model_sweep", int(m.group(1)), int(m.group(2)) / 100.0
    for prefix, cond in PREFIX_CONDITIONS:
        if session_id.startswith(prefix):
            return cond, 8, 1.0
    return "unknown", None, None


def summarize(df):
    rows = []
    for sid, g in df.groupby("session_id"):
        cond, _, fault_mult = classify(sid)
        steps = int(g["step_number"].max())
        n_machines = int(g["machine_id"].nunique())
        valid = ((g["action_taken"] == 0) & (g["is_faulted"] == False)).sum()
        tp_total = valid / steps if steps else float("nan")
        row = {
            "session_id": sid,
            "condition": cond,
            "n_machines": n_machines,
            "fault_multiplier": fault_mult,
            "steps": steps,
            "protocol_matched": steps == EXPECTED_STEPS,
            "throughput_total": round(tp_total, 4),
            "throughput_per_machine": round(tp_total / n_machines, 4),
            "avg_queue_depth": round(g["queue_depth"].mean(), 4),
            "fault_fraction": round((g["is_faulted"] == True).mean(), 4),
        }
        counts = g["action_taken"].value_counts(normalize=True)
        for a, name in ACTION_NAMES.items():
            row[f"pct_{name}"] = round(100 * counts.get(a, 0.0), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["condition", "n_machines", "fault_multiplier", "session_id"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also export full telemetry (large)")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql_query("SELECT * FROM factory_telemetry;", conn)
    conn.close()

    summary = summarize(df)
    summary.to_csv("results/results_per_session.csv", index=False)
    matched = summary[summary["protocol_matched"]]
    print(f"Wrote results/results_per_session.csv: {len(summary)} sessions "
          f"({len(matched)} protocol-matched, {len(summary) - len(matched)} excluded).")
    print(matched.groupby("condition")["throughput_total"].agg(["count", "mean", "std"]).round(3))

    if args.full:
        keep = set(matched["session_id"])
        df[df["session_id"].isin(keep)].drop(columns=["id"]).to_csv(
            "results/telemetry_matched.csv.gz", index=False, compression="gzip")
        print("Wrote results/telemetry_matched.csv.gz")
