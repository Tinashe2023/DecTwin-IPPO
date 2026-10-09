import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box

from envs.factory_env import FactoryTwinEnv


class CentralizedFactoryEnv(gym.Env):
    """
    Single-agent wrapper around FactoryTwinEnv for a CENTRALIZED controller baseline.

    One policy observes the full global state of every machine and outputs a joint
    action for all machines simultaneously -- this is the single-point-of-failure
    architecture the paper's Introduction argues against. Same underlying physics
    (thermal model, Poisson arrivals, fault injection) as FactoryTwinEnv -- only
    observability and control are centralized, so this isolates exactly the
    variable the paper claims matters.
    """

    metadata = {"render_modes": []}

    def __init__(self, mode: str = "training"):
        super().__init__()
        self._env = FactoryTwinEnv(mode=mode)
        self.n_agents = len(self._env.possible_agents)

        # Full global state: [cpu_load, temperature, queue_depth, is_faulted] x n_agents
        self.observation_space = Box(low=0.0, high=150.0, shape=(4 * self.n_agents,), dtype=np.float32)
        # One of 4 discrete actions per machine, chosen jointly by a single policy
        self.action_space = MultiDiscrete([4] * self.n_agents)

    def _global_obs(self):
        return np.concatenate(
            [self._env.factory_state[a] for a in self._env.possible_agents]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        self._env.reset(seed=seed, options=options)
        return self._global_obs(), {}

    def step(self, action):
        actions = {
            agent: int(action[i]) for i, agent in enumerate(self._env.possible_agents)
        }
        _, rewards, terminations, truncations, infos = self._env.step(actions)

        # Single global reward -- the centralized controller optimizes system-wide
        # throughput directly, same quantity the decentralized agents jointly optimize.
        global_reward = float(sum(rewards.values()))
        terminated = any(terminations.values()) if terminations else False
        truncated = any(truncations.values()) if truncations else False

        return self._global_obs(), global_reward, terminated, truncated, {}
