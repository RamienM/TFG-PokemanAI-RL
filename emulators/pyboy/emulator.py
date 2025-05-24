from pyboy import PyBoy
from pyboy.utils import WindowEvent
import numpy as np
import os

class GameEmulator:

    VALID_ACTIONS = {
        0: WindowEvent.PRESS_ARROW_DOWN,
        1: WindowEvent.PRESS_ARROW_LEFT,
        2: WindowEvent.PRESS_ARROW_RIGHT,
        3: WindowEvent.PRESS_ARROW_UP,
        4: WindowEvent.PRESS_BUTTON_A,
        5: WindowEvent.PRESS_BUTTON_B,
        6: WindowEvent.PRESS_BUTTON_START
    }

    REALISE_ACTIONS = {
        WindowEvent.PRESS_ARROW_DOWN : WindowEvent.RELEASE_ARROW_DOWN,
        WindowEvent.PRESS_ARROW_LEFT : WindowEvent.RELEASE_ARROW_LEFT,
        WindowEvent.PRESS_ARROW_RIGHT : WindowEvent.RELEASE_ARROW_RIGHT,
        WindowEvent.PRESS_ARROW_UP : WindowEvent.RELEASE_ARROW_UP,
        WindowEvent.PRESS_BUTTON_A : WindowEvent.RELEASE_BUTTON_A,
        WindowEvent.PRESS_BUTTON_B : WindowEvent.RELEASE_BUTTON_B,
        WindowEvent.PRESS_BUTTON_START : WindowEvent.RELEASE_BUTTON_START
    }
    
    def __init__(self, config):
        rom_path = config["gb_path"]
        window_type = "null" if config["headless"] else "SDL2"
        emulation_speed = (
            6 if "emulation_speed" not in config else config["emulation_speed"]
        )

        self.pyboy = PyBoy(rom_path, window=window_type, sound_emulated=False)
        self.pyboy.set_emulation_speed(emulation_speed)
        
    def step(self,action):
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Acción {action} no válida. Debe estar en {list(self.VALID_ACTIONS.keys())}")
        
       
        button = self.VALID_ACTIONS[action]
        press_step = 8 # As in their paper

        self.pyboy.send_input(button, True)  # Press the button
        self.pyboy.tick(press_step)
        self.pyboy.send_input(self.REALISE_ACTIONS[button], False)  # Realise the button
        self.pyboy.tick(press_step)
        self.pyboy.tick(1)
    
    def load_state(self,initial_state):
        with open(initial_state, "rb") as f:
            self.pyboy.load_state(f)

    def get_ram_state(self):
        return self.pyboy.memory
    
    def get_screen(self):
        return np.expand_dims(self.pyboy.screen.ndarray[:, :, 0], axis=-1)  #(144,160,1)
    
    def close(self):
        self.pyboy.stop()