from src.environments.Pokemon_Red.global_map import local_to_global, GLOBAL_MAP_SHAPE

import uuid
from gymnasium import Env, spaces
from collections import deque
import numpy as np
import json 
import cv2

class PokemonRedEnv(Env):

    def __init__(self, emulator, memory_reader, vision_model, video_recorder, config):
        self.emulator = emulator
        self.memory_reader = memory_reader
        self.vision_model = vision_model
        self.video_recorder = video_recorder
        

        ##--------- Config -----------------
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        
        self.enc_freqs = (
            5 if "fourier_encode_freq" not in config else config["fourier_encode_freq"]
        )
        self.quantity_action_storage = (
            5 if "actions_stack" not in config else config["actions_stack"]
        )
        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        )
        self.max_steps = (
            1000 if "max_steps" not in config else config["max_steps"]
        )
        self.init_state = (
            "../states/STELLE.state" if "init_state" not in config else config["init_state"]
        )
        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        )
        self.coords_pad = (
            12 if "coords_pad" not in config else config["coords_pad"]
        )

        self.step_discount = config.get("step_discount", 0)

        #--------- Observation space config -----------------
        self.recent_actions = deque(maxlen=self.quantity_action_storage)
        self.action_space = spaces.Discrete(len(self.emulator.VALID_ACTIONS))

        self.output_shape_main = (72,80)
        self.observation_space = spaces.Dict(
            {
                "main_screen": spaces.Box(low=0, high=255, shape=self.output_shape_main, dtype=np.uint8),
                "segmented_screen" : spaces.Box(low=0, high=15, shape=self.output_shape_main, dtype=np.uint8),
                "health": spaces.Box(low=0, high=1),
                "badges": spaces.Discrete(8),
                "events": spaces.MultiBinary(self.memory_reader.get_difference_between_events()),
                "map": spaces.Box(low=0, high=255, shape=(self.coords_pad*4,self.coords_pad*4, 1), dtype=np.uint8),
                "visit_map": spaces.Box(low=0.0, high=1.0, shape=(self.coords_pad*4, self.coords_pad*4, 1), dtype=np.float32),
                "recent_actions": spaces.MultiDiscrete([len(self.emulator.VALID_ACTIONS)]*self.quantity_action_storage),
                "remaining_ratio": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "coords": spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.int32),
            }
        )

        self.reset_count = 0

        #--------- Video recorder -----------------
        if self.video_recorder:
            self.video_recorder.start_writing()

        #---------Loading regions -----------------
        regions_path = "src/environments/Pokemon_Red/map_data.json"
        with open(regions_path, "r") as f:
            data = json.load(f)
        self.regions = []
        for r in data["regions"]:
            x0, y0 = r["coordinates"]
            w, h = r["tileSize"]
            # cada región: [x_min, x_max, y_min, y_max, id, name]
            self.regions.append({
                "id": r["id"],
                "name": r["name"],
                "xmin": x0,
                "xmax": x0 + w,
                "ymin": y0,
                "ymax": y0 + h,
            })


    def reset(self, seed=None, options=None):
        self.seed = seed
        self.emulator.load_state(self.init_state)

        self.seen_coords = {}

        self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.uint8)
        self.visit_count_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)

        self.recent_main_screen = np.zeros(self.output_shape_main, dtype=np.uint8)
        self.main_screen = np.zeros(self.output_shape_main, dtype=np.uint8)
        
        self.recent_actions = deque([0] * self.quantity_action_storage, maxlen=self.quantity_action_storage)


        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.region_count_r = 0
        ##Vision
        self.base_event_flags = self.memory_reader.read_events_done()

        self.visited_regions = set()
        self.region_cells_seen = {}  # id_region -> set de (x, y)

        for region in self.regions:
            self.region_cells_seen[region["id"]] = set()
        
        self.reward_region_exploration = [0]*len(self.regions)

        self.progress_reward = self.calculate_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self._get_obs(), {}
        
    def step(self, action):
        self.emulator.step(action)
        self.update_recent_actions(action)
        self.update_seen_coords()
        self.update_explore_map()
        self.update_heal_reward()
        self.update_visit_map()

        self.party_size = self.memory_reader.read_pokemon_in_party()

        new_reward = self.update_reward()
        step_limit_reached = self.check_if_done()
        obs = self._get_obs()
        self.print_info()

        self.step_count += 1
        info = {"episode": {"r": self.total_reward,  "l": self.step_count, "exploration_reward":self.reward_scale * (self.get_exploration_reward()) * 3}}
        return obs, new_reward, False, step_limit_reached, info

    def _get_obs(self):
        screen = self.emulator.get_screen()
        segmentation = self.segmented_screen(screen)

        reduced_screen = self.reduce_screen(screen[:,:,0])
        reduced_segmentation = self.reduce_screen(segmentation)

        x,y,m = self.memory_reader.get_game_coords()
        observation = {
            "main_screen": reduced_screen, 
            "segmented_screen": reduced_segmentation,
            "health": np.array([self.read_hp_fraction()]),
            "badges": self.memory_reader.read_bagdes_in_possesion(),
            "events": np.array(self.memory_reader.read_event_bits(), dtype=np.int8),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions,
            "remaining_ratio":  np.array([self.get_remaining_in_current_region()], dtype=np.float32),
            "coords": np.array([x,y,m], dtype=np.int32),
            "visit_map": self.get_visit_map_crop()[..., None],  # visitas normalizadas
        }
        
        return observation
    
    def check_if_done(self):
        return self.step_count >= self.max_steps - 1

    def print_info(self):
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.5f}"
            prog_string += f" sum: {self.total_reward:5.5f}"
            print(f"\r{prog_string}", end="", flush=True)
    
    def update_recent_actions(self, action):
        self.recent_actions.appendleft(action)
    
    def update_seen_coords(self):
        """
        Registra las coordenadas visitadas, contando las visitas y guardando el paso en que se visitaron.
        
        Parámetros:
        - current_step: Número de paso actual en la simulación.
        """
        if not self.memory_reader.is_in_battle():
            x_pos, y_pos, map_n = self.memory_reader.get_game_coords()
            coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
            

            # Incrementar el número de veces que se ha visitado la coordenada
            if coord_string in self.seen_coords:
                self.seen_coords[coord_string]['count'] += 1
            else:
                self.seen_coords[coord_string] = {'count': 1}
    
    def read_hp_fraction(self):
        hp_sum =  self.memory_reader.get_sum_all_current_hp()
        max_hp_sum = self.memory_reader.get_sum_all_max_hp()

        if max_hp_sum == 0:
            return hp_sum
        
        return hp_sum / max_hp_sum
    
    def get_agent_stats(self):
        x_pos, y_pos, map_n = self.memory_reader.get_game_coords()
        levels = self.memory_reader.get_all_player_pokemon_level()
        reg = self.get_current_region()
        region_id   = reg["id"]   if reg else None
        region_name = reg["name"] if reg else "Unknown"
        region_reward = self.get_region_reward()

        return {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "region_id": region_id,
                "region_name": region_name,
                "region_reward": region_reward,
                "pcount": self.memory_reader.read_pokemon_in_party(),
                "team_p": self.memory_reader.get_all_player_pokemon_name(),
                "levels": levels,
                "levels_sum": sum(levels),
                "hp": self.read_hp_fraction(),
                "coord_count": len(self.seen_coords),
                "explore": self.get_exploration_reward(),
                "deaths": self.died_count,
                "badge": self.memory_reader.read_bagdes_in_possesion(),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
                "step_penality": self.step_count*self.step_discount,
                "action": self.recent_actions[0]
            }
        
    #--------- REWARDS FUNCTIONS -----------------
    def update_reward(self):
        self.progress_reward = self.calculate_reward()

        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )

        if not self.memory_reader.is_in_battle():
            new_total -= self.step_count * self.step_discount
        new_step = new_total - self.total_reward

        self.total_reward = new_total

        return new_step
    
    def calculate_reward(self):
        return {
            "event": self.reward_scale * self.memory_reader.read_events_done(),
            "heal": self.reward_scale * self.total_healing_rew * 0.0001,
            "dead": self.reward_scale * self.died_count / 5,
            "badge": self.reward_scale * self.memory_reader.read_bagdes_in_possesion() * 10,
            "explore": self.reward_scale * self.get_exploration_reward() * 2,
            "region": self.reward_scale * self.get_region_reward()
        }
    
    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.memory_reader.read_pokemon_in_party() == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount * heal_amount
            else:
                self.died_count += 1
        self.last_health = cur_health
    
    def get_exploration_reward(self):
       region = self.get_current_region()
       self.reward_region_exploration[int(region["id"])] = 1-self.get_remaining_in_current_region()
       return sum(self.reward_region_exploration)
    
    def get_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            out = np.zeros((self.coords_pad*2, self.coords_pad*2), dtype=np.uint8)
        else:
            out = self.explore_map[
                c[0]-self.coords_pad:c[0]+self.coords_pad,
                c[1]-self.coords_pad:c[1]+self.coords_pad
            ]
        # einops is better when transformations are more complex
        out = np.repeat(out, 2, axis=0)  
        out = np.repeat(out, 2, axis=1)
        return out
    
    def update_visit_map(self):
        """
        Aumenta en 1 el contador de visitas de la casilla actual en visit_count_map.
        """
        if not self.memory_reader.is_in_battle():
            x, y, m = self.memory_reader.get_game_coords()
            gy, gx = self.get_global_coords()

            if 0 <= gy < self.visit_count_map.shape[0] and 0 <= gx < self.visit_count_map.shape[1]:
                self.visit_count_map[gy, gx] += 1

    def get_visit_map_crop(self):
        """
        Devuelve el recorte del mapa de visitas normalizado (valores entre 0 y 1).
        """
        c = self.get_global_coords()
        crop = np.zeros((self.coords_pad*2, self.coords_pad*2), dtype=np.float32)

        if 0 <= c[0] < self.visit_count_map.shape[0] and 0 <= c[1] < self.visit_count_map.shape[1]:
            crop = self.visit_count_map[
                c[0]-self.coords_pad:c[0]+self.coords_pad,
                c[1]-self.coords_pad:c[1]+self.coords_pad
            ].astype(np.float32)

        # Normalización
        max_val = np.max(self.visit_count_map)
        if max_val > 0:
            crop /= max_val  # ahora entre 0 y 1

        # Ampliamos como haces con explore_map
        crop = np.repeat(crop, 2, axis=0)
        crop = np.repeat(crop, 2, axis=1)

        return crop


    def update_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            print(f"coord out of bounds! global: {c} game: {self.memory_reader.get_game_coords()}")
            pass
        else:
            self.explore_map[c[0], c[1]] = 255
    
    def get_global_coords(self):
        x_pos, y_pos, map_n = self.memory_reader.get_game_coords()
        return local_to_global(y_pos, x_pos, map_n)
    
    def segmented_screen(self,screen):
        if self.video_recorder:
            pred, maps = self.vision_model.predict_with_overlay(screen)
            self.video_recorder.save_videos(screen=screen,maps=maps)
        else:
            pred = self.vision_model.predict(screen)
        return pred
    
    def reduce_screen(self,screen):
        return cv2.resize(screen, (screen.shape[1] // 2, screen.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
    
    
    def get_current_region(self):
        """
        Dado x,y absolutos, devuelve la región que los contiene,
        o None si no está en ninguna.
        """
        _, _, map_n = self.memory_reader.get_game_coords()
        for r in self.regions:
            if int(r["id"]) == int(map_n):
                return r
        return None
    
    def get_region_reward(self):
        """
        Devuelve una recompensa si el agente entra en una nueva región.
        """
        region = self.get_current_region()
        if not region:
            return self.region_count_r 

        region_id = region["id"]

        if region_id not in self.visited_regions:
            self.visited_regions.add(region_id)
            self.region_count_r += 5

        return self.region_count_r
    
    def get_remaining_in_current_region(self):
        region = self.get_current_region()
        x, y, _ = self.memory_reader.get_game_coords()
        
        if not region:
            return 1.0  # No estás en ninguna región, asumimos sin explorar

        region_id = region["id"]

        # Inicializa el set si aún no existe
        if region_id not in self.region_cells_seen:
            self.region_cells_seen[region_id] = set()

        # Marca la celda como visitada
        self.region_cells_seen[region_id].add((x, y))

        # Calcula progreso
        w = region["xmax"] - region["xmin"]
        h = region["ymax"] - region["ymin"]
        total_cells = w * h
        visited_cells = len(self.region_cells_seen[region_id])
        
        # Cálculo del porcentaje explorado
        explored_ratio = visited_cells / total_cells

        # Queremos que: 
        #   - explored_ratio >= 0.75  -> return 0.0
        #   - explored_ratio == 0.0   -> return 1.0
        #   - explored_ratio in (0, 0.75) -> lineal de 1.0 a 0.0

        threshold = 0.75
        if explored_ratio >= threshold:
            return 0.0
        else:
            return 1.0 - (explored_ratio / threshold)

