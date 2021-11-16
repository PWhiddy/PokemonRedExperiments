from pathlib import Path
import torch
from torch.utils.data import Dataset
from pyboy import WindowEvent
from rollout import Rollout

class GBDataset(Dataset):

    def __init__(self, dir_path):     
        self.dir_path = dir_path
        self.all_paths = [p for p in Path(dir_path).glob('*.json')]
    
    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx, remove_last=1900):
        roll = Rollout.from_saved_path(self.all_paths[idx], self.dir_path)
        frames = roll.frames[:-remove_last]
        actions = roll.actions[:-remove_last]
        last_actions = [WindowEvent.PASS] + actions[:-1]
        final_reward = roll.rewards[:-remove_last][-1]
        reward_to_go = [final_reward -  r for r in roll.rewards[:-remove_last]]

        return (
            torch.tensor(frames, dtype=torch.float) / 255.0, 
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(last_actions, dtype=torch.long),
            torch.tensor(reward_to_go, dtype=torch.float)
        )