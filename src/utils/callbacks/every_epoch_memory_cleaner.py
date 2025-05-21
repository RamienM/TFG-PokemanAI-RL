from stable_baselines3.common.callbacks import BaseCallback
import gc
import torch

class EveryEpochMemoryCleaner(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self):
        return True
    
    def _on_training_end(self) -> None:
        print("[MemoryCleaner] Liberando memoria...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[MemoryCleaner] Memoria liberada.")