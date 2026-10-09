# DecTwin-IPPO

Code, trained checkpoints and evaluation tooling for:

> **Fault-Aware Industrial Digital Twins via Cooperative Multi-Agent Reinforcement Learning: A Decentralized
> Partially Observable Markov Decision Process Architecture with Proximal Policy Optimization and Neighborhood-Aware
> Temporal Coordination**. Manuscript submitted to _Engineering Applications of Artificial Intelligence_
> (special issue on Agentic AI).

The repository contains a simulated industrial digital twin (a ring of identical machines with Poisson job
arrivals, thermal dynamics and temperature-dependent stochastic faults) and the scripts used to train and
evaluate decentralized and baseline schedulers on it.

## What is implemented

- **Environment** (`envs/factory_env.py`): PettingZoo `ParallelEnv`; ring of 8 machines by default (configurable),
  200-step episodes, four discrete actions per machine (PROCESS, IDLE, REROUTE_LEFT, REROUTE_RIGHT).
- **Observation**: each agent sees its own telemetry plus the raw telemetry of its left and right neighbors
  (CPU load, temperature, queue depth, fault flag; 12 values in total).
- **Learning algorithm**: Independent PPO with parameter sharing (Ray RLlib 2.10, PyTorch). One shared policy
  network is used by every agent. Each agent is trained on its **own local (per-machine) reward**; there is
  **no centralized critic**.
- **Network** (`models/comm_net.py`): FC(64) -> FC(64) -> GRU(128) -> actor head and critic head. The GRU is a
  temporal memory over each agent's own observation sequence.
- **Not implemented**: learned message passing or attention between agents. The class name `CommNetGRU` and the
  model key `gru_comm_net` are legacy names from an earlier design; they were left unchanged so the released
  checkpoints load without modification. The model contains no communication layer. Likewise the ablation
  files/prefixes `no_comm` / `noComm_` mean "no neighbor observation".

## Environment details (`envs/factory_env.py`)

| Item                                   | Value                                                                               |
| -------------------------------------- | ----------------------------------------------------------------------------------- |
| Job arrivals                           | Poisson, rate 0.25 per machine per step                                             |
| Ambient temperature / cooling constant | 22 C / k = 0.05 (Newton cooling)                                                    |
| Heating when PROCESSing                | CPU ~ N(90, 3) %; temperature += 0.03 x CPU (about 2.7 C per step); capped at 120 C |
| Idle / faulted CPU                     | ~ N(5, 1) %                                                                         |
| Fault probability per step             | (0.01 + 0.005 x max(0, T - 85 C)) x fault multiplier (default 1.0)                  |
| Fault recovery                         | probability 0.15 per step while faulted                                             |
| Standing queue penalty                 | -0.2 x queue depth, every step (`queue_penalty_weight`)                             |
| PROCESS with queue > 0                 | +2.0, queue -1                                                                      |
| PROCESS with empty queue               | -0.5                                                                                |
| PROCESS while faulted                  | -10.0                                                                               |
| Any other action while faulted         | -1.0                                                                                |
| REROUTE (not faulted, queue > 0)       | +0.1, one job moved to the left/right neighbor                                      |

Machines are updated sequentially in a fixed order within each step, so a reroute takes effect on the
neighbor's queue immediately. The centralized baseline (`envs/centralized_env.py`) uses the same physics and is
trained on the sum of the per-machine rewards.

## Conditions evaluated

| Condition                                                              | Script(s)                                                     | Checkpoint folder          |
| ---------------------------------------------------------------------- | ------------------------------------------------------------- | -------------------------- |
| Rule-based baseline                                                    | `baseline_runner.py`                                          | n/a                        |
| Centralized PPO (global state, joint action)                           | `train_centralized.py`, `centralized_runner.py`               | `checkpoints_centralized/` |
| Ablation: no neighbor observation                                      | `train_ablation.py --ablation no_comm`, `eval_ablation.py`    | `checkpoints_no_comm/`     |
| Ablation: no standing queue penalty                                    | `train_ablation.py --ablation no_penalty`, `eval_ablation.py` | `checkpoints_no_penalty/`  |
| Full model                                                             | `train.py`, `marl_runner.py`                                  | `checkpoints/`             |
| Scalability / fault-rate sweeps (full-model checkpoint, no retraining) | `sweep_runner.py`                                             | `checkpoints/`             |

Note on the centralized baseline: it is a single feed-forward policy (two 256-unit tanh layers, no recurrence)
over the concatenated 32-value global state, with a `MultiDiscrete([4]*8)` action space (one categorical head
per machine). The decentralized policies use the GRU network above. Training budget (100 iterations x 4,000
steps, lr 5e-5, 30 SGD epochs, minibatch 128) is the same for all learned conditions.

## Setup

Python 3.11 (the version used for all reported runs). CPU only; no GPU is required.

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows; use `source .venv/bin/activate` elsewhere
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Telemetry is logged to PostgreSQL (default: database `dectwin`, user `postgres`, host `localhost`,
**port 5433**). Create the database and table once, then copy `.env.example` to `.env` and set `DB_PASSWORD`
(the scripts load `.env` automatically; real environment variables also work):

```bash
createdb -U postgres -p 5433 dectwin
psql -U postgres -p 5433 -d dectwin -f database/schema.sql
cp .env.example .env        # then edit DB_PASSWORD   (Windows: copy .env.example .env)
```

Weights & Biases logging in the `train*.py` scripts is optional: set `WANDB_API_KEY` in `.env` to enable it.

## Reproducing the results

| Step | Command                                                                                                                  | Notes                                                |
| ---- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| 1    | `python baseline_runner.py`                                                                                              | repeat 8 times                                       |
| 2    | `python train.py`                                                                                                        | about 5.8 h on one CPU worker; writes `checkpoints/` |
| 3    | `python marl_runner.py`                                                                                                  | repeat 8 times (deterministic actions)               |
| 4    | `python train_centralized.py`, then `python centralized_runner.py`                                                       | repeat the runner 5 times                            |
| 5    | `python train_ablation.py --ablation no_comm` and `--ablation no_penalty`, then `python eval_ablation.py --ablation ...` | repeat each runner 5 times                           |
| 6    | `python sweep_runner.py --n_machines 4` (and 16); `python sweep_runner.py --fault_multiplier 0.5` (and 2.0, 3.0)         | 3 runs per setting                                   |
| 7    | `python aggregate_results.py`, `python analyze_actions.py`, `python analyze_sweep.py`, `python audit_sessions.py`        | tables and statistics                                |
| 8    | `python evaluate_marl.py`                                                                                                | queue-stress figures (paper Figure 3)                |
| 9    | `python export_results.py`                                                                                               | writes `results/results_per_session.csv`             |

To skip training, use the released checkpoints and start from step 3. Checkpoints are RLlib 2.10 cloudpickle
files; only load checkpoints from sources you trust.

`evaluate_marl.py` plots the **most recent** `baseline_` and `run_` session in the database, so make sure those
are 2000-step sessions before using the figure.

## Evaluation protocol

- Every reported session is exactly **2000 steps** (ten consecutive 200-step episodes). Sessions of any other
  length are excluded by `aggregate_results.py` and flagged in `results/results_per_session.csv`.
- Learned policies act deterministically (greedy). The rule-based baseline (`baseline_runner.py`) is stochastic:
  a faulted machine or a machine with an empty queue stays IDLE; otherwise it picks PROCESS (70 %),
  REROUTE_LEFT (15 %) or REROUTE_RIGHT (15 %). It does not look at neighbor state, so it can reroute work
  onto a faulted neighbor.
- **Throughput** is the number of PROCESS actions taken by non-faulted machines per step, summed over machines
  (per-machine values divide by the number of machines). It counts PROCESS actions, not completed jobs; average
  queue depth is reported alongside it for this reason.
- Reported variability is across repeated evaluation rollouts of **one trained policy per condition**.
  Training seeds were not varied.

## Reference results (protocol-matched sessions)

| Condition                         | Throughput (mean +/- std) | n   |
| --------------------------------- | ------------------------- | --- |
| Rule-based baseline               | 1.977 +/- 0.027           | 8   |
| Centralized PPO                   | 3.554 +/- 0.049           | 5   |
| Ablation: no neighbor observation | 6.693 +/- 0.125           | 5   |
| Ablation: no queue penalty        | 2.024 +/- 0.016           | 5   |
| Full model                        | 4.854 +/- 0.146           | 8   |

Per-machine throughput of the full-model checkpoint (trained only at n = 8): 0.631 (n = 4), 0.607 (n = 8),
0.613 (n = 16); and 0.655, 0.607, 0.540, 0.513 at 0.5x, 1x, 2x, 3x the nominal fault rate.

## Hardware used

Windows laptop, 4 physical / 8 logical CPU cores, about 8 GB RAM, no GPU. One 100-iteration training run
(about 400,000 environment steps) took about 5.8 hours.

## Data

- `results/results_per_session.csv`: per-session metrics for all sessions (protocol-matched flag included).
- Full per-step telemetry is not included in this initial code release.

The live dashboard (`inference_server.py` + `index.html`, WebSocket on port 8765) is a demo and was not used for
any reported result. All reported evaluation runners write directly to PostgreSQL.

## License and citation

MIT License (see `LICENSE`). Citation details will be added when they are available.
