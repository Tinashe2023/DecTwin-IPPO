import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium.spaces import Discrete, Box

class FactoryTwinEnv(ParallelEnv):
    """
    Decentralized Digital Twin Environment with Rigorous Engineering Physics.
    Includes M/M/1 Queue dynamics, Thermodynamic modeling, and Stress-induced faults.
    """
    metadata = {"render_modes": ["human"], "name": "dectwin_v0"}

    def __init__(self, mode='training', queue_penalty_weight=0.2, n_machines=8, fault_rate_multiplier=1.0):
        super().__init__()
        self.mode = mode
        self.queue_penalty_weight = queue_penalty_weight
        self.fault_rate_multiplier = fault_rate_multiplier
        self.max_steps = 200
        self.current_step = 0
        
        # --- PHYSICS CONSTANTS ---
        self.ambient_temp = 22.0       # Room temperature in Celsius
        self.max_temp = 120.0          # Critical meltdown temperature
        self.cooling_k = 0.05          # Newton's cooling constant
        self.heat_gen_rate = 0.3       # Temp rise per 1% CPU utilization
        
        # --- QUEUE CONSTANTS (M/M/1) ---
        self.arrival_rate = 0.25       # Lambda (Poisson arrival rate)
        
        self.possible_agents = [f"machine_{i}" for i in range(1, n_machines + 1)]
        self.agents = self.possible_agents[:]

        self.action_spaces = {
            agent: Discrete(4) 
            for agent in self.possible_agents
        }

        # Observation Space: Own State (4) + Left Neighbor (4) + Right Neighbor (4) = 12
        # State variables: [cpu_load, temperature, queue_depth, is_faulted]
        self.observation_spaces = {
            agent: Box(low=0.0, high=150.0, shape=(12,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Internal State tracking
        self.factory_state = {}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_step = 0
        
        # Initialize physical states: [cpu_load, temperature, queue_depth, is_faulted]
        self.factory_state = {
            agent: np.array([0.0, self.ambient_temp, 0.0, 0.0], dtype=np.float32)
            for agent in self.possible_agents
        }

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def observe(self, agent):
        """Ring Topology: Gather state from self and nearest neighbors."""
        agent_idx = self.possible_agents.index(agent)
        num_agents = len(self.possible_agents)
        
        left_idx = (agent_idx - 1) % num_agents
        right_idx = (agent_idx + 1) % num_agents
        
        left_agent = self.possible_agents[left_idx]
        right_agent = self.possible_agents[right_idx]
        
        own_state = self.factory_state[agent]
        left_state = self.factory_state[left_agent]
        right_state = self.factory_state[right_agent]
        
        return np.concatenate([own_state, left_state, right_state]).astype(np.float32)

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        self.current_step += 1
        is_done = self.current_step >= self.max_steps

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        for agent in self.agents:
            action = actions.get(agent)
            state = self.factory_state[agent]
            
            # --- 1. DYNAMIC ENVIRONMENT UPDATES ---
            if self.mode == 'training' or self.mode == 'inference':
                # Poisson Job Arrivals
                arrivals = np.random.poisson(self.arrival_rate)
                state[2] += arrivals
                
                # Thermodynamics: Newton's Law of Cooling
                # T_new = T_ambient + (T_current - T_ambient) * e^(-k * dt) + Heat_Generated
                dt = 1.0 # time step
                cooling_effect = (state[1] - self.ambient_temp) * (1.0 - np.exp(-self.cooling_k * dt))
                state[1] -= cooling_effect

                # Stress-Induced Faults (Probability scales with temperature)
                # Base fault rate + exponential penalty above 85C
                fault_prob = 0.01
                if state[1] > 85.0:
                    fault_prob += 0.005 * (state[1] - 85.0)
                fault_prob *= self.fault_rate_multiplier
                
                if np.random.rand() < fault_prob:
                    state[3] = 1.0
                elif state[3] == 1.0 and np.random.rand() < 0.15:
                    state[3] = 0.0 # Auto-recovery

           # --- 2. AGENT ACTION PHYSICS & REWARDS ---
            reward = 0.0
            is_faulted = (state[3] == 1.0)
            queue_depth = state[2]
            
            # Queue penalty scaled by the configurable environment weight
            reward -= (queue_depth * self.queue_penalty_weight)

            if action == 0 and not is_faulted:
                # Processing: Spike CPU, Generate Heat
                state[0] = min(100.0, np.random.normal(90.0, 3.0))
                state[1] = min(self.max_temp, state[1] + (state[0] * self.heat_gen_rate * 0.1))
                
                if queue_depth > 0:
                    reward += 2.0  # Big reward for actually clearing a job
                    state[2] = max(0, queue_depth - 1)
                else:
                    reward -= 0.5 # Wasting cycles
                    
            else:
                # Idle/Faulted/Rerouting: CPU drops to baseline
                state[0] = max(0.0, np.random.normal(5.0, 1.0))
                
                if is_faulted:
                    if action == 0:
                        reward -= 10.0 # Fatal error: running broken machine
                    else:
                        reward -= 1.0  # Minor downtime penalty
                else:
                    if action in [2, 3] and queue_depth > 0:
                        reward += 0.1 # Reroute reward
                        state[2] -= 1
                        
                        # Apply physical reroute to neighbors
                        idx = self.possible_agents.index(agent)
                        neighbor_idx = (idx - 1) if action == 2 else (idx + 1)
                        neighbor_idx = neighbor_idx % len(self.possible_agents)
                        self.factory_state[self.possible_agents[neighbor_idx]][2] += 1

            rewards[agent] = float(reward)
            observations[agent] = self.observe(agent)
            terminations[agent] = False
            truncations[agent] = is_done
            infos[agent] = {}

        if is_done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos