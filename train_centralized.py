import os
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from envs.centralized_env import CentralizedFactoryEnv
from env_config import load_project_env

load_project_env()


def env_creator(config):
    return CentralizedFactoryEnv(mode="training")


if __name__ == "__main__":
    print("Initializing Ray...")
    ray.init(ignore_reinit_error=True, object_store_memory=200 * 1024 * 1024, _memory=2 * 1024 * 1024 * 1024)

    env_name = "dectwin_centralized_v0"
    register_env(env_name, env_creator)

    print("Configuring Centralized PPO (single policy, full global observability)...")

    # Same core PPO hyperparameters as the decentralized run (train.py) --
    # keeping these identical means any performance gap reflects the
    # architecture (centralized vs. decentralized), not a hyperparameter
    # difference.
    config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch")
        .rollouts(num_rollout_workers=1)
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=30,
            gamma=0.99,
            lr=5e-5,
            clip_param=0.3,
            kl_coeff=0.2,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
            },
        )
    )

    algo = config.build()

    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints_centralized")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # WandB logging is opt-in via env var -- no hardcoded key, and this
    # still trains fine with just console output if you skip it.
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb
        wandb.init(project="DecTwin-IPPO", name="Centralized_Baseline_Run_1", config=config.to_dict())
    else:
        print("[i] WANDB_API_KEY not set -- training with console logging only.")

    print("\nStarting Centralized Baseline Training Loop (100 Iterations)...")
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

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")
        algo.save(checkpoint_dir)
    finally:
        algo.stop()
        ray.shutdown()
        if use_wandb:
            wandb.finish()
        print("Ray shutdown complete.")
