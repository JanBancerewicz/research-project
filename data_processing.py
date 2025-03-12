import pandas as pd


def save_data(data, filename):
    df = pd.DataFrame(data, columns=["Timestamp", "Heart Rate"])

    df["Time"] = pd.to_datetime(df["Timestamp"], unit="s")

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Data saved in {filename}")
