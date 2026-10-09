import numpy as np
from gymnasium.spaces import Box

from envs.factory_env import FactoryTwinEnv


class NoCommFactoryEnv(FactoryTwinEnv):
    """
    Ablation: removes neighbor-state sharing entirely. Each agent sees only its
    OWN state -- no left/right neighbor information. Isolates whether the
    neighbor-state observation actually contributes to performance, versus the
    temporal GRU memory and the shared reward structure alone accounting for the gains.
    (File/class names keep the legacy "no_comm" label so released checkpoints and
    session prefixes stay valid; no communication layer exists in any model.)

    Same underlying physics as FactoryTwinEnv (inherited, untouched) -- only
    the observation function changes.
    """

    def __init__(self, mode: str = "training"):
        super().__init__(mode=mode)
        # Own state only: [cpu_load, temperature, queue_depth, is_faulted]
        self.observation_spaces = {
            agent: Box(low=0.0, high=150.0, shape=(4,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def observe(self, agent):
        return self.factory_state[agent].astype(np.float32)
