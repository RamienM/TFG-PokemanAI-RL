from stable_baselines3.common.callbacks import BaseCallback
import os

class RewardThresholdCallback(BaseCallback):
    """
    Callback que detiene el entrenamiento si la recompensa total de un episodio
    cae por debajo de un umbral definido, y guarda el modelo.
    """

    def __init__(self, reward_threshold=-100, verbose=0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        # Revisamos cada info para detectar episodios que cumplan la condición
        for info in infos:
            if "episode" in info:
                total_reward = info["episode"]["r"]

                if total_reward <= self.reward_threshold:
                    return False

        return True  # Continuar entrenando
