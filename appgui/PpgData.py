import asyncio
import websockets
from queue import Queue
import threading

import pandas as pd
from appgui.data import DataProducerThread


class PpgData(threading.Thread):
    def __init__(self, data_queue, stop_event, name="EcgDataFile"):
        super().__init__()
        self.data_queue = data_queue

    def run(self):
        # Startuje serwer WebSocket i zwraca wiadomość z kolejki, jeśli jakaś przyszła
        async def handler(websocket):
            print("🔌 Connected to client")
            try:
                async for message in websocket:
                    f = message.split(' ')
                    self.data_queue.put(float(f[1]))  # oryginalne działanie
            except websockets.exceptions.ConnectionClosed:
                print("❌ Connection closed")

        async def start_server():
            # server = await websockets.serve(handler, "localhost", 8765)
            server = await websockets.serve(handler, "0.0.0.0", 8765)
            print("✅ Server is listening on port 8765")
            await server.wait_closed()

        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start_server())
            loop.run_forever()

        # Uruchom serwer w osobnym wątku tylko raz
        if not hasattr(self, 'ws_thread_started'):
            threading.Thread(target=run_server, daemon=True).start()


