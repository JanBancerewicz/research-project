import asyncio

import numpy as np
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
import matplotlib.pyplot as plt
import struct

POLAR_NAME = "Polar H10 D222AF24"
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"

async def scan():
    """Scan for BLE devices."""
    devs = await BleakScanner.discover()
    addr = ""
    for dev in devs:
        print(f"{dev.name}: {dev.address}")
        if dev.name == POLAR_NAME:
            addr = dev.address
    return addr


def notification_handler_wrapper(ecg):
    return lambda sender, data: notification_handler(sender, data, ecg)


def pmd_control_handler(characteristic: BleakGATTCharacteristic, data: bytearray):
    hex = [f"{i:02x}" for i in data]
    print(f"CTRL: {hex}")


def notification_handler(characteristic: BleakGATTCharacteristic, data: bytearray, ecg):
    hex = [f"{i:02x}" for i in data]
    print(f"DATA: {hex}")
    if data[0] == 0x00:  # 0x00 = ECG
        i = 9
        frame_type = data[i]
        if frame_type == 0:  # 0 = ECG Data
            i += 1
            while len(data[i:][0:3]) == 3:
                ecg.append(int.from_bytes(data[i:][0:2], byteorder='little', signed=True))
                i += 3


async def connect(address):
    async with BleakClient(address) as client:
        print(f"Connected: {client.is_connected}")
        ecg = []
        await client.start_notify(PMD_CONTROL, pmd_control_handler)
        await client.start_notify(PMD_DATA, notification_handler_wrapper(ecg))

        await client.write_gatt_char(PMD_CONTROL,
                                     bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0e, 0x00]))

        await asyncio.sleep(60*30)

        await client.write_gatt_char(PMD_CONTROL, bytearray([0x03, 0x00]))

        await client.stop_notify(PMD_DATA)
        await client.stop_notify(PMD_CONTROL)
    return ecg


async def get_data():
    """Get data from Polar H10."""
    address = await scan()
    ecg = await connect(address)
    n = len(ecg)
    time = np.arange(0, n * (100 / 13), 100 / 13)
    data = []
    for i in range(n):
        data.append((time[i], ecg[i]))
    return data


plt.show()
