import pandas as pd
import psycopg2

from env_config import get_db_params

DB_PARAMS = get_db_params()

if not DB_PARAMS.get("password"):
    raise RuntimeError(
        "Database password is not configured. Set DB_PASSWORD in the environment or .env file."
    )

EXPECTED_STEPS = 2000
PREFIXES = ("baseline_", "centralized_", "run_")


def load_data():
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql_query("SELECT session_id, step_number FROM factory_telemetry;", conn)
    conn.close()
    return df


if __name__ == "__main__":
    print(f"[*] Auditing every session against the expected protocol ({EXPECTED_STEPS} steps)...")
    df = load_data()
    max_steps_per_session = df.groupby("session_id")["step_number"].max().sort_index()

    groups = {p: [] for p in PREFIXES}
    groups["other"] = []

    for sid, max_step in max_steps_per_session.items():
        matched_prefix = next((p for p in PREFIXES if sid.startswith(p)), "other")
        groups[matched_prefix].append((sid, int(max_step)))

    total_clean, total_dirty = 0, 0
    for prefix, items in groups.items():
        if not items:
            continue
        print(f"\n=== {prefix} ===")
        clean, dirty = 0, 0
        for sid, max_step in sorted(items):
            if max_step == EXPECTED_STEPS:
                flag = "OK"
                clean += 1
            else:
                flag = f"MISMATCH -- expected {EXPECTED_STEPS}, got {max_step}"
                dirty += 1
            print(f"  {sid}: {max_step} steps  [{flag}]")
        print(f"  --> {clean} clean / {dirty} mismatched")
        total_clean += clean
        total_dirty += dirty

    print(f"\n[*] TOTAL: {total_clean} clean sessions, {total_dirty} mismatched sessions across all groups.")
    if total_dirty:
        print("[!] Mismatched sessions should be excluded from any reported mean/std --")
        print("    they weren't run under the same fixed-length protocol as the rest.")
