-- Run once against the database used by the scripts (default name: dectwin, port 5433).
CREATE TABLE IF NOT EXISTS factory_telemetry (
    id           SERIAL PRIMARY KEY,
    session_id   VARCHAR(50),
    step_number  INT,
    machine_id   VARCHAR(20),
    cpu_load     FLOAT,
    temperature  FLOAT,
    queue_depth  INT,
    is_faulted   BOOLEAN,
    action_taken INT,
    timestamp    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_factory_telemetry_session ON factory_telemetry (session_id);
