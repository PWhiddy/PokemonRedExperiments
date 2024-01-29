import asyncio
import websockets
import json

import gymnasium as gym

X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
MAP_N_ADDRESS = 0xD35E

class StreamWrapper(gym.Wrapper):
    def __init__(self, env, stream_metadata={}):
        super().__init__(env)
        self.ws_address = "wss://poke-ws-test-ulsjzjzwpa-ue.a.run.app/broadcast"
        self.stream_metadata = stream_metadata
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = self.loop.run_until_complete(
                websockets.connect(self.ws_address)
        )
        self.upload_interval = 200
        self.steam_step_counter = 0
        self.coord_list = []
        self.emulator = env.pyboy if env.pyboy is not None else env.game

    def step(self, action):

        x_pos = self.emulator.get_memory_value(X_POS_ADDRESS)
        y_pos = self.emulator.get_memory_value(Y_POS_ADDRESS)
        map_n = self.emulator.get_memory_value(MAP_N_ADDRESS)
        self.coord_list.append([x_pos, y_pos, map_n])

        if self.steam_step_counter >= self.upload_interval:
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

    async def broadcast_ws_message(self, message):
        try:
            await self.websocket.send(message)
        except websockets.exceptions.WebSocketException as e:
            # attempt reconnection after a timeout or error
            self.websocket = await websockets.connect(self.ws_address)