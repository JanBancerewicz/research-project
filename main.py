import asyncio
import data_processing as dp
import polar
import peaks

DATA_FILENAME = "data\\Natalia1.csv"


async def main():
    #data = await polar.get_data()
   # dp.save_data(data, DATA_FILENAME)
    peaks.plot_all(DATA_FILENAME)

asyncio.run(main())




