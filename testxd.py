import asyncio
import websockets
import random

async def send_random_data():
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to server")
            while True:
                value = random.randint(0, 2000)
                await websocket.send(str(value))
                print(f"📤 Sent: {value}")
                await asyncio.sleep(0.1)  # 100 ms
    except Exception as e:
        print(f"❌ Connection failed or lost: {e}")

if __name__ == "__main__":
    asyncio.run(send_random_data())