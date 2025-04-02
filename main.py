import asyncio

import numpy as np
import pandas as pd
import numeric.plot as p
import polar
from numeric.save_rr import extract_r_indexes, save_csv
from torch import nn
DATA_FILENAME = "data/ECG6.csv"
R_FILENAME = "data/R6.csv"
RUN_POLAR = True

async def main():
   if RUN_POLAR:
      data = await polar.get_data()
      df = pd.DataFrame(data, columns=["timestamp", "ecg"])
      df.to_csv(DATA_FILENAME, index=False)



   data = pd.read_csv(DATA_FILENAME)
   r = extract_r_indexes(data)
   save_csv(r, R_FILENAME)


   p.plot(data)


asyncio.run(main())




