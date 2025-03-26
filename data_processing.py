import pandas as pd
import ast

def save_data(data, filename):
    df = pd.DataFrame(data, columns=["time", "heart rate", "R-R"])

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Data saved in {filename}")


def load_data(filename):
    data = pd.read_csv(filename)
    d = data["R-R"]
    v = []
    for rec in d:
         v.append(ast.literal_eval(rec))
    data["R-R"] = v
    return data

def combine_intervals(data):
    intervals = []
    for i in data["R-R"]:
        for j in i:
            intervals.append(j)
    return intervals