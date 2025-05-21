import uuid
from gymnasium import Env, spaces
from collections import deque
import math
import numpy as np
import os
import json 
from src.utils.video_recorder import VideoRecorder

from src.environments.Pokemon_Red.global_map import local_to_global, GLOBAL_MAP_SHAPE

class PokemonRedEnv(Env):

    def __init__(self, emulator, memory_reader, vision_model, config):
        self.emulator = emulator
        self.memory_reader = memory_reader
        self.vision_model = vision_model
        

        #Config
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.save_video = config["save_video"]
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


        self.recent_actions = deque(maxlen=self.quantity_action_storage)
        self.action_space = spaces.Discrete(len(self.emulator.VALID_ACTIONS))

        self.output_shape_main = (144,160,3)
        self.observation_space = spaces.Dict(
            {
                "main_screen" : spaces.Box(low=0, high=255, shape=self.output_shape_main, dtype=np.uint8),
                "health": spaces.Box(low=0, high=1),
                "badges": spaces.Discrete(8),
                "events": spaces.MultiBinary(self.memory_reader.get_difference_between_events()),
                "map": spaces.Box(low=0, high=255, shape=(
                    self.coords_pad*4,self.coords_pad*4, 1), dtype=np.uint8),
                "recent_actions": spaces.MultiDiscrete([len(self.emulator.VALID_ACTIONS)]*self.quantity_action_storage),
                "nearby": spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(5*5,), dtype=np.int32),
                "seen_summary": spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
            }
        )

        self.reset_count = 0
        if self.save_video:
            self.video_recorder = VideoRecorder()
            self.video_recorder.start_writing()

                # Carga las regiones
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
        # Para rastrear visitas
        self.visited_regions = set()
        self.region_visit_reward = 10
    
    def reset(self, seed=None, options=None):
        self.seed = seed
        self.emulator.load_state(self.init_state)

        self.seen_coords = {}

        self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.uint8)

        self.recent_main_screen = np.zeros(self.output_shape_main, dtype=np.uint8)
        self.main_screen = np.zeros(self.output_shape_main, dtype=np.uint8)
        
        self.recent_actions = deque([0] * self.quantity_action_storage, maxlen=self.quantity_action_storage)

        self.max_event_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.stuck = 0
        self.step_penalty = 0
        self.last_coord = ""
        self.seen_coords_r = 0
        self.menu = False
        self.region_count_r = 0
        ##Vision
        self.base_event_flags = self.memory_reader.read_events_done()

        self.current_event_flags_set = {}

        #self.max_map_progress = 0
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

        if self.memory_reader.is_in_battle():
            self.step_penalty += 0.001
        else:
            self.step_penalty += 0.005

        self.party_size = self.memory_reader.read_pokemon_in_party()

        new_reward = self.update_reward()
        step_limit_reached = self.check_if_done()
        obs = self._get_obs()
        self.print_info()

        self.step_count += 1
        info = {"episode": {"r": self.total_reward,  "l": self.step_count, "exploration_reward":self.reward_scale * (self.get_exploration_reward()) * 3}}
        return obs, new_reward, False, step_limit_reached, info

    def _get_obs(self):
        screen = self.segmented_screen(self.emulator.get_screen())
        observation = {
            "main_screen": screen, 
            "health": np.array([self.read_hp_fraction()]),
            "badges": self.memory_reader.read_bagdes_in_possesion(),
            "events": np.array(self.memory_reader.read_event_bits(), dtype=np.int8),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions,
            "nearby": self.get_nearby(),
            "seen_summary": np.array(self.get_seen_coords_summary(), dtype=np.float32)

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
                self.seen_coords[coord_string] = {'count': 1, 'stuck': 0}
                self.seen_coords_r+= 1
    
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
                "coords_summary": self.get_seen_coords_summary(),
                #"max_map_progress": self.max_map_progress,
                "pcount": self.memory_reader.read_pokemon_in_party(),
                "team_p": self.memory_reader.get_all_player_pokemon_name(),
                "levels": levels,
                "levels_sum": sum(levels),
                "hp": self.read_hp_fraction(),
                "coord_count": len(self.seen_coords),
                "explore": self.get_exploration_reward(),
                "stuck": self.stuck,
                "deaths": self.died_count,
                "badge": self.memory_reader.read_bagdes_in_possesion(),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
                "step_penality": self.step_penalty
            }
        
    

    
    ## REWARD FUNCTIONS

    def calculate_reward(self):
        x, y, m = self.memory_reader.get_game_coords()
        gy, gx = local_to_global(x, y, m)
        return {
            "event": self.reward_scale * self.memory_reader.read_events_done() * 4,
            "heal": self.reward_scale * self.total_healing_rew * 0.01,
            "dead": self.reward_scale * self.died_count / 2,
            "badge": self.reward_scale * self.memory_reader.read_bagdes_in_possesion() * 10,
            "explore": self.reward_scale * self.get_exploration_reward(),
            #"stuck": self.reward_scale * self.get_stuck_penalty(),
            "region": self.reward_scale * self.get_region_reward() ,
            "step_penalty": self.reward_scale * self.step_penalty * -1
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

    def update_reward(self):
        self.progress_reward = self.calculate_reward()

        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )
        new_step = new_total - self.total_reward

        self.total_reward = new_total

        return new_step
    
    def get_exploration_reward(self):

       return self.seen_coords_r / 5
    
    def get_stuck_penalty(self):
        """
        Penalización por estar atascado, considerando visitas frecuentes y recientes a una misma casilla.
        - Se reinicia si han pasado más de 1000 pasos desde la última visita.
        """
        if self.memory_reader.is_in_battle():
            return self.stuck
        x_pos, y_pos, map_n = self.memory_reader.get_game_coords()
        coord_key = f"x:{x_pos} y:{y_pos} m:{map_n}"
        
        if coord_key not in self.seen_coords:
            return self.stuck

        data = self.seen_coords[coord_key]
        count = data['count']
        stuck = data['stuck']
        # Calculo de diferencia
        stuck_penalty = self.compute_stuck_penalty(count)
        data['stuck'] = stuck_penalty
        if stuck_penalty == stuck: 
            new_stuck = stuck
        else:
            new_stuck = stuck_penalty - stuck

        self.stuck += new_stuck
        return self.stuck
    
    def compute_stuck_penalty(self, count, clip=300):
        """
        Calcula una penalización por estar atascado basada en visitas frecuentes y recientes.

        Parámetros:
        - count: número de visitas a la casilla.
        - steps_since_last: pasos desde la última visita.
        - clip: valor máximo de visitas consideradas (default: 300).
        - decay: control del decaimiento temporal (default: 50.0).

        Retorna:
        - Un valor negativo que representa la penalización.
        """
        # Limita el número de visitas y normaliza
        clipped_count = max(0, min(count, clip))
        normalized_count = clipped_count / clip


        # Penalización final con curva cúbica
        penalty = (normalized_count ** 2)
        return -1 * (penalty ** 3)
    
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
    
    def update_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            print(f"coord out of bounds! global: {c} game: {self.memory_reader.get_game_coords()}")
            pass
        else:
            self.explore_map[c[0], c[1]] = 255
    
    # def update_map_progress(self):
    #     map_idx = self.memory_reader.read_progress_map()
    #     map_progress = self.essential_map_locations.get(map_idx, -1)
    #     self.max_map_progress = max(self.max_map_progress, map_progress)
    
    def get_global_coords(self):
        x_pos, y_pos, map_n = self.memory_reader.get_game_coords()
        return local_to_global(y_pos, x_pos, map_n)
    
    def segmented_screen(self,screen):
        maps = self.vision_model.predict(screen)
        if self.save_video:
                self.video_recorder.save_videos(screen=screen,maps=maps)
        return maps
    
    def get_nearby(self):
        """
        Actualiza el estado del agente según la última acción y devuelve
        una lista con el conteo de visitas para las posiciones cercanas en un radio 3 (7x7).

        - Detecta si el agente está en menú.
        - Actualiza coordenadas vistas y conteos para evitar volver a casillas problemáticas.
        - Llama a get_nearby_counts_7x7 para obtener la matriz de conteos.

        Returns:
            list[int]: Lista con 49 valores (7x7) de visitas a casillas alrededor del agente.
        """

        x, y, m = self.memory_reader.get_game_coords()

        # Prevenir que el agente intente siempre volver a una casilla desconocida si fue teletransportado
        action_to_offset = {
            0: (1, 0),    # Abajo → mirar arriba
            1: (0, 1),    # Izquierda → mirar derecha
            2: (0, -1),   # Derecha → mirar izquierda
            3: (-1, 0),   # Arriba → mirar abajo
        }

        last_action = self.recent_actions[0]
        actual = f"x:{x} y:{y} m:{m}"
        if last_action == 6:
            self.menu = True
        if self.menu and not (actual == self.last_coord):
            self.menu = False 
        
        if self.last_coord == actual and last_action < 4 and not self.menu: ## Ha intentado moverse contra un pared
            if last_action in action_to_offset:
                dx, dy = action_to_offset[last_action]
                coord = f"x:{x + dx} y:{y + dy} m:{m}"
                if coord not in self.seen_coords:
                    self.seen_coords[coord] = {'count': np.iinfo(np.int32).max, 'stuck': 0}
        elif last_action < 4 and not self.menu: ## Posible teletransporte
            if last_action in action_to_offset:
                if self.last_coord not in self.seen_coords:
                    self.seen_coords[self.last_coord] = {'count': 1, 'stuck': 0}
                else:
                    self.seen_coords[self.last_coord]['count'] += 1

        self.last_coord = actual

        return self.get_nearby_counts_5x5()
    
    def get_nearby_counts_5x5(self):
        """
        Devuelve una lista con los contadores de visitas para una ventana 5x5
        centrada en la posición actual del agente (radio 2).

        La matriz cubre desde (x+2, y-2) (arriba izquierda) hasta (x-2, y+2) (abajo derecha).

        Returns:
            list[int]: Lista de 25 enteros (5x5) con los conteos de visitas a casillas.
        """
        x, y, m = self.memory_reader.get_game_coords()

        nearby = []
        for dx in range(2, -3, -1):  # de +2 (arriba) a -2 (abajo)
            for dy in range(-2, 3):  # de -2 (izquierda) a +2 (derecha)
                coord = f"x:{x + dx} y:{y + dy} m:{m}"
                count = self.seen_coords.get(coord, {'count': 0})['count']
                nearby.append(count)

        return nearby
    
    def get_seen_coords_summary(self):
        """
        Devuelve un resumen de las coordenadas vistas por el agente:
        - total de coordenadas únicas vistas
        - media del número de visitas por coordenada
        - máximo número de visitas a una misma coordenada
        """
        if not self.seen_coords:
            return [0.0, 0.0, 0.0]
        
        counts = [v['count'] for v in self.seen_coords.values()]
        num_coords_seen = len(counts)
        avg_count = np.mean(counts)
        max_count = np.max(counts)

        return [num_coords_seen, avg_count, max_count]
    
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
            return self.region_count_r  # No estás en ninguna región definida

        region_id = region["id"]  # <- Extraes solo el ID

        if region_id not in self.visited_regions:
            self.visited_regions.add(region_id)
            self.region_count_r += self.region_visit_reward  # recompensa por nueva región

        return self.region_count_r
