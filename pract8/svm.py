import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC

def split_data(df):
    print(df.head())

def main():
    df = pd.read_csv("./dataset_svm.csv")
    split_data(df)
    df_columns = df.columns
    print(df[df_columns].dtypes)
main()