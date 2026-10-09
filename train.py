import os
import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.models import ModelCatalog
from models.comm_net import CommNetGRU 
from envs.factory_env import FactoryTwinEnv
from env_config import load_project_env

load_project_env()

def env_creator(config):
    env = FactoryTwinEnv(mode='training')
    return ParallelPettingZooEnv(env)

if __name__ == "__main__":
    print("Initializing Ray...")
# To this:
    # Notice the underscore before _memory
    ray.init(ignore_reinit_error=True, object_store_memory=200 * 1024 * 1024, _memory=2 * 1024 * 1024 * 1024)
    env_name = "dectwin_v0"
    register_env(env_name, env_creator)

    ModelCatalog.register_custom_model("gru_comm_net", CommNetGRU)

    print("Configuring Independent PPO (parameter-shared GRU policy)...")
    
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
            model={
                "custom_model": "gru_comm_net",
                "max_seq_len": 20, 
            }
        )
    )

    algo = config.build()
    
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Weights & Biases logging is optional: enabled only if WANDB_API_KEY is set.
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ["WANDB_START_METHOD"] = "thread"  # avoids WinError 64 on Windows
        import wandb
        wandb.init(
            project="DecTwin-IPPO",
            entity=None,
            name="GRU_Physics_Run_1",
            config=config.to_dict()
        )
    else:
        print("[i] WANDB_API_KEY not set -- training with console logging only.")

    print("\nStarting Deep Training Loop (100 Iterations)...")
    
    try:
        for i in range(1, 101):
            result = algo.train()
            mean_reward = result.get('episode_reward_mean', 'N/A')
            total_episodes = result.get('episodes_total', 0)
            
            print(f"Iteration {i}: Mean Reward = {mean_reward} | Total Episodes = {total_episodes}")
            
            if use_wandb:
                wandb.log({
                    "iteration": i,
                    "episode_reward_mean": mean_reward,
                    "episodes_total": total_episodes,
                    "policy_loss": result.get('info', {}).get('learner', {}).get('shared_policy', {}).get('learner_stats', {}).get('policy_loss', 0),
                    "vf_loss": result.get('info', {}).get('learner', {}).get('shared_policy', {}).get('learner_stats', {}).get('vf_loss', 0)
                })
            
            if i % 20 == 0 or i == 100:
                save_result = algo.save(checkpoint_dir)
                print(f"\n[💾 CHECKPOINT SAVED] AI Brain backed up to: {save_result}\n")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")
        algo.save(checkpoint_dir)
    finally:
        algo.stop()
        ray.shutdown()
        if use_wandb:
            wandb.finish()  # close the cloud connection safely
        print("Ray shutdown complete.")