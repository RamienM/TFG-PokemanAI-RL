# Source imports
from src.environments.Pokemon_Red.env import PokemonRedEnv
from src.agents.ppo_agent import PPOAgent
from src.utils.stream_agent_wrapper import StreamWrapper
from src.utils.video_recorder import VideoRecorder

from src.utils.callbacks.tensorboard_callback import TensorboardCallback
from src.utils.callbacks.every_epoch_memory_cleaner import EveryEpochMemoryCleaner
from src.utils.callbacks.agent_stats_logger_callback import AgentStatsLoggerCallback
from src.utils.callbacks.reward_threshold_callback import RewardThresholdCallback


# Emulator imports
from emulators.pyboy.emulator import GameEmulator
from emulators.pyboy.memory_reader import MemoryReader
from models.segmentation.STELLE_Pokemon_Segmentation.inference.inferencer import STELLEInferencer

#Queue imports
from server.shared_inferencer import SharedInferencer
from server.inference_server import inference_process


#Stable baselines 3 imports
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed

#Weight & Bias imports
import wandb
from wandb.integration.sb3 import WandbCallback

#Others imports
import yaml
import os
import sys
import json
from datetime import datetime
import multiprocessing as mp

def make_env(rank,session_path, config,  request_q, response_q, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        emulator = GameEmulator(config)
        memory_reader = MemoryReader(emulator)
        if config.get("server", False):
            print(f"[ENV - {rank}] Usando modelo de visión compartido")
            vision_model = SharedInferencer(request_q, response_q)
        else:
            print(f"[ENV - {rank}] Usando modelo de visión propio")
            vision_model = STELLEInferencer()
        video_recorder = None
        if config.get("save_video",False):
            save_path = os.path.join(session_path,"videos",f"env_{rank}")
            os.makedirs(save_path, exist_ok=True)
            video_recorder = VideoRecorder(save_path=save_path)
        env = StreamWrapper(
            PokemonRedEnv(emulator,memory_reader, vision_model,video_recorder,config), 
            stream_metadata = { # All of this is part is optional
                "user": "Ramien", # choose your own username
                "env_id": rank, # environment identifier
                "color": "#7A378B", # choose your color :)
                "extra": "STELLE", # any extra text you put here will be displayed,
                "sprite_id": 4 ## Prueba
            }
        )
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True) 

    # --------- Obteining configuration -----------------
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    
    num_cpu = config.get("num_cpu", 1)

    checkpoint_from_yaml =config.get("load_checkpoint", "")
    checkpoint_arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Priorizar argumento si existe
    file_name = checkpoint_arg if checkpoint_arg else checkpoint_from_yaml

    ep_length = config.get("max_steps", 1000000) * num_cpu
    train_steps_batch = config.get("train_steps_batch", ep_length // 1024) 
    batch_size = config.get("batch_size", ep_length//256)

    iterations = config.get("iterations", 1)
    progress_bar = config.get("progress_bar",False)

    server_enabled = config.get("server", False)
    use_wandb_logging = config.get("use_wandb", True)

    
    # --------- Creating paths and directories -----------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sess_id = f"poke_{timestamp}"

    root_dir = config.get("root_dir","game&results/Pokemon_red_env/checkpoints")
    base_dir = os.path.join(root_dir, sess_id)
    sess_path_logs = os.path.join(base_dir, "tensorboard")  # logs para TensorBoard
    agent_stats_dir = os.path.join(base_dir, "agent_stats")  # CSVs

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(sess_path_logs, exist_ok=True)
    os.makedirs(agent_stats_dir, exist_ok=True)

    # --------- Starting queues -----------------

    if server_enabled:
        manager = mp.Manager()
        request_q = manager.Queue()
        
        response_queues = [manager.Queue() for _ in range(num_cpu)]
        inference_proc = mp.Process(
            target=inference_process,
            args=(request_q,),
        )
        inference_proc.start()
        env = SubprocVecEnv([make_env(i, base_dir, config,  request_q, response_queues[i]) for i in range(num_cpu)])
    else:
        env = SubprocVecEnv([make_env(i,base_dir,  config,  None, None) for i in range(num_cpu)])

    # --------- General training information -----------------
    metadata = {
        "session_id": sess_id,
        "checkpoint_freq": config.get("checkpoint_save_freq", 64),
        "num_envs": num_cpu,
        "ep_length": ep_length,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "notes": config.get("notes", "")  # si agregas esto a tu config.yaml
    }

    metadata_path = os.path.join(base_dir, "run_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    # --------- Preparing Callbacks -----------------
    
    callbacks = [
        CheckpointCallback(save_freq=config.get("checkpoint_save_freq", 64),save_path=base_dir,name_prefix=sess_id), 
        TensorboardCallback(sess_path_logs,num_cpu), 
        EveryEpochMemoryCleaner(),
        AgentStatsLoggerCallback(base_dir=agent_stats_dir,num_envs=num_cpu,save_every=config.get("save_stats_every",50)),
        RewardThresholdCallback(reward_threshold=config.get("reward_threshold",-100))
    ]

    ## --------- Starting Weight&Bias -----------------
    if use_wandb_logging:
        wandb.tensorboard.patch(root_logdir=str(sess_path_logs))
        run = wandb.init(
            project=config.get("wandb_project", "pokemon-train"),
            id=sess_id,
            name=sess_id,
            config=config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
            reinit=True
        )
        callbacks.append(WandbCallback())


    # --------- Init agent model -----------------
    model = PPOAgent(env=env, policy="MultiInputPolicy", verbose=1, n_steps=train_steps_batch, batch_size=batch_size, n_epochs=1, gamma=0.997, ent_coef=0.01, tensorboard_log=sess_path_logs)
    if os.path.exists(file_name + ".zip"):
        print("[TRAIN] Cargando punto de guardado...")
        model.load(path=file_name, env=env)  # Cargar el modelo desde el checkpoint
        print(f"[TRAIN] Punto de guardado {file_name}.zip cargado.")
    
    # --------- Training -----------------

    print(f"[TRAIN] Iniciando el entrenamiento")
    model.train(total_timesteps=ep_length , callbacks=CallbackList(callbacks),progress_bar=progress_bar)
    print(f"[TRAIN] Se ha finalizado el entrenamiento ")
        
    # --------- Closing enviroments, queues and weight&bias -----------------
    env.close()

    last_checkpoint_path = os.path.join(base_dir, "final")

    ## --------- Saving model -----------------
    model.save(last_checkpoint_path)
    with open("last_checkpoint_path.txt", "w") as f:
        f.write(last_checkpoint_path)

    if server_enabled:
        request_q.put((None, None, None)) 
        inference_proc.join()

    if use_wandb_logging:
        run.finish()