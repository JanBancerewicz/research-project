import asyncio
import numpy as np
import pandas as pd
import numeric.plot as p
import polar
from numeric.save_rr import extract_r_indexes, save_csv


DATA_FILENAME = "data/g2_ECG0.csv"
R_FILENAME = "data/R_7.csv"
RUN_POLAR = True
#True for long measure
LONG_MEASURE_b = False

async def long_measure(times, label):
   for i in range(times):
       await get_data(f"data/{label}_ECG{i}.csv", f"data/{label}_R{i}.csv")
       await asyncio.sleep(10)

async def get_data(file, file_r):
   if RUN_POLAR:
      data = await polar.get_data()
      df = pd.DataFrame(data, columns=["timestamp", "ecg"])
      df.to_csv(file, index=False)

   data = pd.read_csv(file)
   r = extract_r_indexes(data)
   save_csv(r, file_r)
   return data, r

async def main():

   long_measure_b = True
   # lm = input("Long Measure? (y/n): ")
   # if lm == "y":
   #    long_measure_b = True


   if long_measure_b:
      # times = 1 is 30 minutes
      times = int(input("Number of times to measure (30 minutes periods): "))
      label = input("File label: ")
      await long_measure(times, label)
   else:
      data = pd.read_csv(DATA_FILENAME)

      p.plot(data)



asyncio.run(main())




