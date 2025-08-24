import asyncio
import threading
from queue import Queue
import time  # <-- Add import

from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic

POLAR_NAME = "Polar H10 D222AF24"  # Or your specific device name
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"



class EcgDataBluetooth(threading.Thread):
    def __init__(self, data_queue, stop_event, name="EcgDataBluetooth"):
        super().__init__(name=name)
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.loop = None
        self.lock = threading.Lock()
        self.device_address = None

    async def scan_for_device(self):
        devices = await BleakScanner.discover()
        for dev in devices:
            if dev.name == POLAR_NAME:
                return dev.address
        raise Exception(f"Device '{POLAR_NAME}' not found.")

    def handle_ecg_data(self, _: BleakGATTCharacteristic, data: bytearray):
        if data[0] == 0x00:
            i = 9
            frame_type = data[i]
            if frame_type == 0:
                i += 1
                timestamp = time.time() * 1000 - (1.0/130.0)*1000
                while len(data[i:]) >= 3:
                    ecg_sample = int.from_bytes(data[i:i+2], byteorder='little', signed=True)
                     # UNIX timestamp
                    timestamp += (1/130)*1000  # Increment timestamp for each sample
                    with self.lock:
                        self.data_queue.put((timestamp, ecg_sample))  # Put tuple (timestamp, sample)
                    i += 3

    async def connect_and_stream(self, address: str):
        async with BleakClient(address) as client:
            print("🔗 Connected to", address)
            await client.start_notify(PMD_DATA, self.handle_ecg_data)
            await client.write_gatt_char(
                PMD_CONTROL,
                bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0e, 0x00])
            )

            try:
                while not self.stop_event.is_set():
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            finally:
                await client.write_gatt_char(PMD_CONTROL, bytearray([0x03, 0x00]))
                await client.stop_notify(PMD_DATA)
                print("🛑 Disconnected")

    def run_async(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.device_address = self.loop.run_until_complete(self.scan_for_device())
            self.loop.run_until_complete(self.connect_and_stream(self.device_address))
        except Exception as e:
            print(f"⚠️ Error: {e}")
        finally:
            self.loop.close()

    def run(self):
        # Run the BLE client in its own thread's event loop
        self.run_async()
