import asyncio
import pandas as pd

import data_processing as dp
import r_peaks
import polar

DATA_FILENAME = "data/ECG3.csv"


async def main():
   data = await polar.get_data()
   dp.save_data(data, DATA_FILENAME)
   data = pd.read_csv(DATA_FILENAME)
   r_peaks.plot(data)


asyncio.run(main())




