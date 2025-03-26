import asyncio
import time

import numpy as np
from bleak import BleakScanner, BleakClient
from matplotlib.animation import FuncAnimation

import R
import matplotlib.pyplot as plt

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

def notification_handler_wrapper(heart_rate_data, line):
    return lambda sender, data: notification_handler(sender, data,heart_rate_data,line)

def notification_handler(sender, data, heart_rate_data, line):
    """Handle incoming heart rate data and save with timestamp."""

    heart_rate = data[1]  # Heart rate in BPM
    rr_intervals = []
    timestamp = time.time()
    if data[0] & 0b00010000:  # Check if RR intervals exist
        for i in range(2, len(data), 2):
            rr_interval = int.from_bytes(data[i:i + 2], byteorder="little") / 1024  # Convert to seconds
            heart_rate_data.append(rr_interval)
            rr_intervals.append(rr_interval)
        if len(heart_rate_data) >= 15:
            breathing_rate = R.calculate_breathing_rate_realtime(heart_rate_data)
            print(f"Częstotliwość oddechowa: {breathing_rate:.2f} oddechów na minutę")
            heart_rate_data = heart_rate_data[5:]  # Wyczyść dane po obliczeniu
    line.append((timestamp, heart_rate, rr_intervals))

    print(f"❤️ Heart Rate: {heart_rate} bpm | ⏱️ R-R Intervals: {rr_intervals}")
    #heart_rate_data.append((timestamp, rr_intervals, heart_rate))




async def connect_to_polar(address):
    """Connect to Polar H10 and receive heart rate data."""
    print("Connecting to Ostap...")
    heart_rate_data = []
    line = []
    async with BleakClient(address) as client:
        if client.is_connected:
            print(f"Connected to Ostap ({address})")
            await client.start_notify(HEART_RATE_UUID, notification_handler_wrapper(heart_rate_data, line))
            await asyncio.sleep(60)  # Receive data for 60 seconds
            await client.stop_notify(HEART_RATE_UUID)
            plt.show()



            return line
        else:
            print("Ostap is broken!")



async def get_data():
    """Get data from Polar H10."""
    address = await scan()
    return await connect_to_polar(address)
