import asyncio
import time
from bleak import BleakScanner, BleakClient


HEART_RATE_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
POLAR_NAME = "Polar H10 D222AF24"

async def scan():
    """Scan for BLE devices."""
    devs = await BleakScanner.discover()
    addr = ""
    for dev in devs:
        print(f"{dev.name}: {dev.address}")
        if dev.name == POLAR_NAME:
            addr = dev.address
    return addr



def notification_handler_wrapper(heart_rate_data):
    return lambda sender, data: notification_handler(sender, data,heart_rate_data)

def notification_handler(sender, data, heart_rate_data):
    """Handle incoming heart rate data and save with timestamp."""
    heart_rate = data[1]
    timestamp = time.time()
    heart_rate_data.append((timestamp, heart_rate))
    print(f"{time.strftime('%H:%M:%S')} - Heart rate: {heart_rate} BPM")




async def connect_to_polar(address):
    """Connect to Polar H10 and receive heart rate data."""
    print("Connecting to Ostap...")
    async with BleakClient(address) as client:
        if client.is_connected:
            heart_rate_data = []
            print(f"Connected to Ostap ({address})")
            await client.start_notify(HEART_RATE_UUID, notification_handler_wrapper(heart_rate_data))
            await asyncio.sleep(60)  # Receive data for 60 seconds
            await client.stop_notify(HEART_RATE_UUID)
            return heart_rate_data
        else:
            print("Ostap is broken!")



async def get_data():
    """Get data from Polar H10."""
    address = await scan()
    return await connect_to_polar(address)
