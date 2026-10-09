import torch
import torch.nn as nn
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

class CommNetGRU(RecurrentNetwork, nn.Module):
    """
    Shared recurrent actor-critic used by every agent (parameter sharing).
    The GRU gives each agent a memory of its own observation sequence. The 12-value
    observation already contains the raw telemetry of the two ring neighbors; there is
    NO message passing or communication layer. The class name "CommNetGRU" and the model
    key "gru_comm_net" are legacy names kept so released checkpoints load unchanged.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # 1. Feature Extractor: processes the raw observation, which is own + left + right
        # neighbor telemetry (cpu_load, temperature, queue_depth, is_faulted) = 12 values
        # (4 values in the "own state only" ablation)
        self.fc1 = nn.Linear(obs_space.shape[0], 64)
        self.fc2 = nn.Linear(64, 64)

        # 2. The GRU core: temporal memory over the agent's own observation sequence
        self.gru_hidden_size = 128
        self.gru = nn.GRU(
            input_size=64, 
            hidden_size=self.gru_hidden_size, 
            batch_first=True
        )

        # 3. Actor Head: Decides the action (PROCESS, IDLE, REROUTE_A, REROUTE_B)
        self.actor_head = nn.Linear(self.gru_hidden_size, num_outputs)

        # 4. Critic Head: Predicts the expected reward to evaluate the action
        self.value_head = nn.Linear(self.gru_hidden_size, 1)

        # Internal state to hold the current value prediction
        self._cur_value = None

    def get_initial_state(self):
        """Initializes the GRU memory with absolute zero at the start of an episode."""
        return [torch.zeros(self.gru_hidden_size, dtype=torch.float32)]

    def forward_rnn(self, inputs, state, seq_lens):
        """The core forward pass of the neural network."""
        # 1. Extract features from the current observation
        x = torch.relu(self.fc1(inputs))
        x = torch.relu(self.fc2(x))

        # 2. Pass features AND the previous memory state through the GRU
        # PyTorch GRU expects state shape: [num_layers, batch_size, hidden_size]
        h_in = state[0].unsqueeze(0) 
        gru_out, h_out = self.gru(x, h_in)

        # 3. Calculate Value (Critic)
        self._cur_value = self.value_head(gru_out).squeeze(-1)

        # 4. Calculate Action Logits (Actor)
        logits = self.actor_head(gru_out)

        # Return the action logits and the new memory state
        return logits, [h_out.squeeze(0)]

    def value_function(self):
        """Returns the critic's evaluation for RLlib's PPO algorithm."""
        return torch.reshape(self._cur_value, [-1])