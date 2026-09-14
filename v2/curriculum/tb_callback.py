"""
tb_callback.py -- a TensorBoard callback that works for ANY number of envs.

The repo's original ``tensorboard_callback.TensorboardCallback`` hard-codes an
explore-map image grid with ``rearrange(..., "(r f) h w -> (r h) (f w)", r=2)``,
which requires an *even* env count (the baseline always ran 64). That crashes for
1 env (visual mode) or any odd count.

This drop-in replacement keeps the same logging (per-env stat means/maxes, the
explore-map heatmaps, event-flag dump) but chooses the image grid rows safely based
on the actual env count, and additionally logs the curriculum scalars. The original
file is left untouched for backwards compatibility.
"""

import json
import os

import numpy as np
from einops import rearrange, reduce
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter


def _merge_dicts(dicts):
    sum_dict, count_dict, distrib_dict = {}, {}, {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)):
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1
                distrib_dict.setdefault(k, []).append(v)
    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]
        distrib_dict[k] = np.array(distrib_dict[k])
    return mean_dict, distrib_dict


def _grid_rows(n: int) -> int:
    """Pick a row count that evenly divides ``n`` (so rearrange never fails)."""
    if n <= 1:
        return 1
    for r in (4, 3, 2):
        if n % r == 0:
            return r
    return 1


class CurriculumTensorboardCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None

    def _on_training_start(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(str(self.log_dir), "histogram"))

    def _on_step(self) -> bool:
        # only do the (relatively heavy) logging at episode boundaries of env 0
        if self.training_env.env_method("check_if_done", indices=[0])[0]:
            all_infos = self.training_env.get_attr("agent_stats")
            all_final_infos = [stats[-1] for stats in all_infos if stats]
            if all_final_infos:
                mean_infos, distributions = _merge_dicts(all_final_infos)
                for key, val in mean_infos.items():
                    self.logger.record(f"env_stats/{key}", val)
                for key, distrib in distributions.items():
                    self.writer.add_histogram(f"env_stats_distribs/{key}", distrib, self.n_calls)
                    self.logger.record(f"env_stats_max/{key}", float(max(distrib)))

            explore_map = np.array(self.training_env.get_attr("explore_map"))
            map_sum = reduce(explore_map, "f h w -> h w", "max")
            self.logger.record("trajectory/explore_sum", Image(map_sum, "HW"),
                               exclude=("stdout", "log", "json", "csv"))

            n = explore_map.shape[0]
            rows = _grid_rows(n)
            if n % rows == 0 and rows > 0:
                map_row = rearrange(explore_map, "(r f) h w -> (r h) (f w)", r=rows)
                self.logger.record("trajectory/explore_map", Image(map_row, "HW"),
                                   exclude=("stdout", "log", "json", "csv"))

            list_of_flag_dicts = self.training_env.get_attr("current_event_flags_set")
            merged_flags = {k: v for d in list_of_flag_dicts for k, v in d.items()}
            self.logger.record("trajectory/all_flags", json.dumps(merged_flags))

            # --- adaptive reward controller + intrinsic + subgoal (from env 0 info) ---
            infos = self.locals.get("infos", [])
            if infos:
                adapt = infos[0].get("adaptive_stats", {}) or {}
                for key, val in adapt.items():
                    if isinstance(val, (int, float)):
                        self.logger.record(f"adaptive/{key}", float(val))
                tgt = infos[0].get("subgoal_target_map")
                if tgt is not None:
                    self.logger.record("planner/subgoal_target_map", float(tgt))

        return True

    def _on_training_end(self):
        if self.writer:
            self.writer.close()
