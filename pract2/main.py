import pandas as pd
import numpy as np

def main():
    print("reaading data")

    df = pd.read_table("data/amesHousing.txt")

    print(f"read correctly row:{len(df)} and columns: {len(df.columns)} ")
    print(df.head())



if __name__ == "__main__":
    main()