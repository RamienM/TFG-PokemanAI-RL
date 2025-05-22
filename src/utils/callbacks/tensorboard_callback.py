import os
import json

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from einops import rearrange, reduce

def merge_dicts(dicts):
    sum_dict = {}
    count_dict = {}
    distrib_dict = {}

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

class TensorboardCallback(BaseCallback):

    def __init__(self, log_dir, num_envs, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self.num_envs = num_envs
        self.step_counters = [0] * num_envs

    def _on_training_start(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'histogram'))

    def _on_step(self) -> bool:
        envs = self.model.get_env()
        dicts = []
    
        for env_idx in range(self.num_envs):
            dicts.append(envs.env_method("get_agent_stats", indices=env_idx)[0])

        # Calcular promedios y distribuciones
        mean_infos, distributions = merge_dicts(dicts)

        for key, val in mean_infos.items():
            self.logger.record(f"env_stats/{key}", val)
            self.writer.add_scalar(f"env_stats/{key}", val, self.num_timesteps)

        for key, distrib in distributions.items():
            self.writer.add_histogram(f"env_stats_distribs/{key}", distrib, self.num_timesteps)
            self.logger.record(f"env_stats_max/{key}", max(distrib))

        return True


    
    def _on_training_end(self):
        # Guardar mapas de exploración al final del entrenamiento
        explore_map = np.array(self.training_env.get_attr("explore_map"))
        map_sum = reduce(explore_map, "f h w -> h w", "max")
        self.logger.record("trajectory/explore_sum", Image(map_sum, "HW"), exclude=("stdout", "log", "json", "csv"))

        if explore_map.shape[0] == 1:
            map_row = explore_map.squeeze(0)
        else:
            map_row = rearrange(explore_map, "(r f) h w -> (r h) (f w)", r=2)

        self.logger.record("trajectory/explore_map", Image(map_row, "HW"), exclude=("stdout", "log", "json", "csv"))

        # Dump final para asegurar que se guardan
        self.logger.dump(self.n_calls)

        if self.writer:
            self.writer.close()


