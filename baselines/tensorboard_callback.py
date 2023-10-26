from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
import numpy as np
from einops import rearrange

def merge_dicts_by_mean(dicts):
    sum_dict = {}
    count_dict = {}

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)): 
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]

    return mean_dict

class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        
        if self.training_env.env_method("check_if_done", indices=[0])[0]:
            all_infos = self.training_env.get_attr("agent_stats")
            all_final_infos = [stats[-1] for stats in all_infos]
            mean_infos = merge_dicts_by_mean(all_final_infos)
            for key,val in mean_infos.items():
                self.logger.record(f"env_stats/{key}", val)
            
            images = self.training_env.env_method("render") # use reduce_res=False for full res screens
            images_arr = np.array(images)
            images_row = rearrange(images_arr, "b h w c -> h (b w) c")
            self.logger.record("trajectory/image", Image(images_row, "HWC"), exclude=("stdout", "log", "json", "csv"))

        return True

