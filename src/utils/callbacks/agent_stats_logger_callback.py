from stable_baselines3.common.callbacks import BaseCallback
import os
import json

class AgentStatsLoggerCallback(BaseCallback):
    """
    Callback que guarda self.agent_stats[] de cada hilo en archivos JSONL separados, escribiendo cada N pasos.
    """

    def __init__(self, base_dir="agent_stats_logs", num_envs=1, save_every=1, verbose=0):
        super().__init__(verbose)
        self.base_dir = base_dir
        self.num_envs = num_envs
        self.save_every = save_every
        self.files = []
        self.step_counters = [0] * num_envs  # pasos acumulados por env

    def _on_training_start(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        for i in range(self.num_envs):
            env_dir = os.path.join(self.base_dir, f"env_{i}")
            os.makedirs(env_dir, exist_ok=True)
            jsonl_path = os.path.join(env_dir, "agent_stats.jsonl")

            f = open(jsonl_path, "a", encoding="utf-8")
            self.files.append(f)

    def _on_step(self) -> bool:
        envs = self.model.get_env()

        for env_idx in range(self.num_envs):
            self.step_counters[env_idx] += 1

            if self.step_counters[env_idx] >= self.save_every:
                stats_list = envs.env_method("get_agent_stats", indices=env_idx)[0]
                if stats_list:
                    f = self.files[env_idx]
                    json.dump(stats_list, f, default=str)
                    f.write("\n")  
                self.step_counters[env_idx] = 0  # reinicia el contador

        return True

    def _on_training_end(self) -> None:
        for f in self.files:
            f.close()
