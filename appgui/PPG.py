from appgui.data import DataProducerThread


import asyncio
import threading
import websockets
from queue import Queue

class PPGDATA(DataProducerThread):
    def __init__(self, queue, port=8765):
        DataProducerThread.__init__(self, )
        self.port = port
        self.dane_queue = Queue()
        self.latest_value = None
        self.lock = threading.Lock()

        # Start WebSocket server in background thread
        self.server_thread = threading.Thread(target=self.run_websocket_server)
        self.server_thread.daemon = True
        self.server_thread.start()

    def signal_func(self, counter):
        # Zwraca ostatnią odebraną wartość jako int
        with self.lock:
            return self.latest_value

    def run_websocket_server(self):
        async def handler(websocket):
            print("🔌 Connected to client")
            try:
                async for message in websocket:
                    try:
                        value = int(message)
                        with self.lock:
                            self.latest_value = value
                        print(f"📥 Received int: {value}")
                    except ValueError:
                        print(f"⚠️ Invalid int: {message}")
            except websockets.exceptions.ConnectionClosed:
                print("❌ Connection closed")

        async def start():
            server = await websockets.serve(handler, "0.0.0.0", self.port)
            print(f"✅ Server is listening on port {self.port}")
            await server.wait_closed()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start())
        loop.run_forever()
