import os
import argparse
import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

from env_config import load_project_env
from models.comm_net import CommNetGRU
from envs.factory_env import FactoryTwinEnv
from envs.no_comm_env import NoCommFactoryEnv

load_project_env()

ABLATIONS = {
    "no_comm": {
        "env_class": NoCommFactoryEnv,
        "env_kwargs": {},
        "checkpoint_dir": "checkpoints_no_comm",
        "description": "Neighbor-state observation removed (own state only, 4-dim obs)",
    },
    "no_penalty": {
        "env_class": FactoryTwinEnv,
        "env_kwargs": {"queue_penalty_weight": 0.0},
        "checkpoint_dir": "checkpoints_no_penalty",
        "description": "Standing Queue Penalty removed (weight 0 instead of 0.2)",
    },
}


def make_env_creator(spec):
    def env_creator(config):
        env = spec["env_class"](mode="training", **spec["env_kwargs"])
        return ParallelPettingZooEnv(env)
    return env_creator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True, choices=list(ABLATIONS.keys()))
    args = parser.parse_args()

    spec = ABLATIONS[args.ablation]
    print(f"[*] Running ablation: {args.ablation} -- {spec['description']}")

    ray.init(ignore_reinit_error=True, object_store_memory=200 * 1024 * 1024, _memory=2 * 1024 * 1024 * 1024)

    env_name = f"dectwin_{args.ablation}_v0"
    register_env(env_name, make_env_creator(spec))
    ModelCatalog.register_custom_model("gru_comm_net", CommNetGRU)

    # Same core PPO hyperparameters as the main run (train.py) and the
    # centralized baseline -- only the environment/observation changes,
    # so any performance gap reflects the ablated component, not a
    # hyperparameter difference.
    config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch")
        .rollouts(num_rollout_workers=1)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        )
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=30,
            gamma=0.99,
            lr=5e-5,
            clip_param=0.3,
            kl_coeff=0.2,
            model={
                "custom_model": "gru_comm_net",
                "max_seq_len": 20,
            },
        )
    )

    algo = config.build()

    checkpoint_dir = os.path.join(os.getcwd(), spec["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)

    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb
        wandb.init(project="DecTwin-IPPO", name=f"Ablation_{args.ablation}", config=config.to_dict())
    else:
        print("[i] WANDB_API_KEY not set -- training with console logging only.")

    print(f"\nStarting '{args.ablation}' ablation training (100 iterations)...")
    try:
        for i in range(1, 101):
            result = algo.train()
            mean_reward = result.get("episode_reward_mean", "N/A")
            total_episodes = result.get("episodes_total", 0)
            print(f"Iteration {i}: Mean Reward = {mean_reward} | Total Episodes = {total_episodes}")

            if use_wandb:
                wandb.log({
                    "iteration": i,
                    "episode_reward_mean": mean_reward,
                    "episodes_total": total_episodes,
                })

            if i % 20 == 0 or i == 100:
                save_result = algo.save(checkpoint_dir)
                print(f"\n[CHECKPOINT SAVED] {save_result}\n")

                if use_wandb and wandb.run is not None:
                    checkpoint = getattr(save_result, "checkpoint", save_result)
                    checkpoint_path = getattr(checkpoint, "path", None)

                    if checkpoint_path and os.path.isdir(checkpoint_path):
                        artifact = wandb.Artifact(
                            name=f"ablation-{args.ablation}-checkpoint",
                            type="model",
                            metadata={"iteration": i},
                        )
                        artifact.add_dir(checkpoint_path)
                        wandb.run.log_artifact(
                            artifact,
                            aliases=["latest", f"iteration-{i}"],
                        )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")
        algo.save(checkpoint_dir)
    finally:
        algo.stop()
        ray.shutdown()
        if use_wandb:
            wandb.finish()
        print("Ray shutdown complete.")
