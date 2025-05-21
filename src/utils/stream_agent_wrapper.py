import asyncio
import websockets
import json

import gymnasium as gym

X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
MAP_N_ADDRESS = 0xD35E

class StreamWrapper(gym.Wrapper):
    def __init__(self, env, stream_metadata={}):
        super().__init__(env)
        self.ws_address = "wss://transdimensional.xyz/broadcast"
        self.stream_metadata = stream_metadata
        self.loop = asyncio.new_event_loop()
        #SE ESTABA CREANDO MUCHAS CONEXIONES!!
        if not asyncio.get_event_loop().is_running():
            self.loop = asyncio.get_event_loop()
        else:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        self.websocket = None
        self.loop.run_until_complete(
            self.establish_wc_connection()
        )
        self.upload_interval = 150
        self.steam_step_counter = 0
        self.env = env
        self.coord_list = []
        if hasattr(env, "emulator"):
            self.emulator = env.emulator
        elif hasattr(env, "game"):
            self.emulator = env.game
        else:
            raise Exception("Could not find emulator!")

    def step(self, action):

        x_pos = self.emulator.get_ram_state()[X_POS_ADDRESS]
        y_pos = self.emulator.get_ram_state()[Y_POS_ADDRESS]
        map_n = self.emulator.get_ram_state()[MAP_N_ADDRESS]
        self.coord_list.append([x_pos, y_pos, map_n])

        if self.steam_step_counter >= self.upload_interval:
            #self.stream_metadata["extra"] = f"coords: {len(self.env.seen_coords)}"
            self.loop.run_until_complete(
                self.broadcast_ws_message(
                    json.dumps(
                        {
                          "metadata": self.stream_metadata,
                          "coords": self.coord_list
                        }
                    )
                )
            )
            self.steam_step_counter = 0
            self.coord_list = []

        self.steam_step_counter += 1

        return self.env.step(action)
    
    #NO SE CERRABA DEJANDO MEMORIA MUERTA
    def close(self):
        try:
            if self.websocket is not None:
                self.loop.run_until_complete(self.websocket.close())
            if self.loop.is_running():
                self.loop.stop()
            self.loop.close()
        except Exception as e:
            print("Error cerrando WebSocket:", e)

        super().close()  # Llama al método close del entorno original

    async def broadcast_ws_message(self, message):
        if self.websocket is None:
            await self.establish_wc_connection()
        if self.websocket is not None:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.WebSocketException as e:
                self.websocket = None

    async def establish_wc_connection(self):
        try:
            self.websocket = await websockets.connect(self.ws_address)
        except:
            self.websocket = None
