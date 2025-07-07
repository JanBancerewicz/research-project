import asyncio
import websockets
from queue import Queue
import threading

import pandas as pd
from appgui.data import DataProducerThread


class PpgData(DataProducerThread):
    def __init__(self, data_queue, stop_event, name="EcgDataFile"):
        super().__init__(data_queue, stop_event, self.signal_func, name)
        self.message_queue = Queue()  # przechowuje wiadomości odebrane przez WebSocket

    def signal_func(self, counter):
        # Startuje serwer WebSocket i zwraca wiadomość z kolejki, jeśli jakaś przyszła
        async def handler(websocket):
            print("🔌 Connected to client")
            try:
                async for message in websocket:
                    print(f"📨 Received message: {message}")
                    self.message_queue.put(message)
                    self.data_queue.put(message)  # oryginalne działanie
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
            self.ws_thread_started = True
            threading.Thread(target=run_server, daemon=True).start()

        # Teraz czekaj, aż jakaś wiadomość pojawi się w kolejce i zwróć ją
        try:
            message = self.message_queue.get(timeout=1.0)  # czekaj max 1s
            return message
        except:
            return None  # brak danych jeszcze
