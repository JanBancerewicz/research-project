import asyncio

import matplotlib.pyplot as plt
import numpy as np

import data_processing as dp
import peaks
import polar
import pandas as pdresearch
import RR
import R
import plot
DATA_FILENAME = "data\\Johny4.csv"


async def main():
    #data = await polar.get_data()
   # dp.save_data(data, DATA_FILENAME)
    data = dp.load_data(DATA_FILENAME)
    rr = dp.combine_intervals(data)
    plt.figure()
    plt.plot(data["heart rate"])
    v = []
    t = np.linspace(0, 60, len(rr))
    for r in rr:
        v.append(60/r )
    plt.plot(t, v)
    plt.title("HR from rr")
    plt.show()
    #peaks.plot_all(DATA_FILENAME)
    #RR.calculate_breathing_rate_from_RR(rr)
asyncio.run(main())




