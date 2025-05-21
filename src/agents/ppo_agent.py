from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
import torch

class PPOAgent:
    def __init__(self, env, policy="MlpPolicy", verbose=1, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01, tensorboard_log=None):
        """
        Inicializa el modelo PPO con los parámetros especificados.

        :param env: El entorno donde se va a entrenar el modelo.
        :param policy: La política que se utilizará en PPO (por defecto 'MlpPolicy').
        :param verbose: Nivel de verbosidad (por defecto 1).
        :param n_steps: El número de pasos en cada actualización (por defecto 2048).
        :param batch_size: Tamaño del batch en cada actualización (por defecto 64).
        :param n_epochs: Número de épocas para cada actualización (por defecto 10).
        :param gamma: Factor de descuento (por defecto 0.99).
        :param ent_coef: Coeficiente de la entropía (por defecto 0.01).
        :param tensorboard_log: Directorio para logs de TensorBoard (por defecto None).
        """
        # Inicializa el modelo PPO con los parámetros dados
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[PPO] Usando el dispositivo {device}")
        self.model = PPO(policy, env, verbose=verbose, n_steps=n_steps, batch_size=batch_size, 
                         n_epochs=n_epochs, gamma=gamma, ent_coef=ent_coef, tensorboard_log=tensorboard_log, device=device)

    def train(self, total_timesteps, callbacks=None, tb_log_name= "poke_ppo", progress_bar=False):
        """
        Método de entrenamiento del agente PPO.
        :param total_timesteps: Número total de pasos de entrenamiento.
        :param callbacks: Lista de callbacks (pueden ser utilizados para logging, monitoreo, etc.)
        :param tb_log_name: Nombre para el log de TensorBoard.
        """
        # Usamos CallbackList para aceptar múltiples callbacks si es necesario.
        if callbacks is not None and not isinstance(callbacks, list):
            callbacks = [callbacks]

        # Usamos CallbackList para aceptar múltiples callbacks si es necesario.
        callback_list = CallbackList(callbacks) if callbacks else None
        self.model.learn(total_timesteps=total_timesteps, callback=callback_list,tb_log_name=tb_log_name, reset_num_timesteps=False,progress_bar=progress_bar)

    def save(self, path="models/ppo/ppo_agent"):
        """
        Guarda el modelo PPO en el directorio especificado.

        :param path: Ruta para guardar el modelo.
        """
        self.model.save(path)

    def load(self, path="models/ppo/ppo_agent", env = None):
        """
        Carga el modelo PPO desde el archivo especificado.

        :param path: Ruta donde se encuentra el modelo guardado.
        """
        self.model = PPO.load(path,env)
    

    def predict(self, obs, deterministic=True):
        """
        Realiza una predicción con el modelo PPO dado un estado de observación.

        :param obs: La observación del entorno.
        :param deterministic: Si la predicción es determinista (por defecto True).
        """
        return self.model.predict(obs, deterministic=deterministic)
