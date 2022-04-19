from pathlib import Path
import torch
import torch.nn.functional as F
from pyboy import WindowEvent
from rnn_model import RNNModel

class RNNAgent:

    def __init__(self, state_dict_path, desired_return, temp=1.0):
        self.name = Path(state_dict_path).stem
        self.desired_return = desired_return
        self.model = RNNModel(vis_emb_dim=32, hidden_dim=16)
        self.model.load_state_dict(torch.load(state_dict_path))
        self.model.eval()
        self.temp = temp
        self.prevent_suicide = False
        self.current_hidden = None

    def get_name(self):
        return f'rnn action agent {self.name}'

    def reset(self):
        self.current_hidden = None

    def get_action(self, latest_state, rollout):
        remaining_desired_reward = self.desired_return
        # subtract reward earned so far
        if len(rollout.rewards) > 0:
            remaining_desired_reward -= rollout.rewards[-1]

        prev_action = WindowEvent.PASS
        if len(rollout.actions) > 0:
            prev_action = rollout.actions[-1]

        with torch.no_grad():
            action_logits, hidden_state = self.model(
                to_torch_batch(latest_state, torch.float), 
                to_torch_batch(remaining_desired_reward, torch.float),
                to_torch_batch(prev_action, torch.long), 
                prev_hidden=self.current_hidden)
            
            self.current_hidden = hidden_state
            
            # sample from action distribution
            logits = action_logits[0][0]
            tempered_logits = logits / self.temp
            probs = F.softmax(tempered_logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()

        if self.prevent_suicide and action == WindowEvent.QUIT:
            action = WindowEvent.PASS
        return action

def to_torch_batch(t, dtype):
    return torch.unsqueeze(torch.unsqueeze(torch.tensor(t, dtype=dtype), 0), 0)