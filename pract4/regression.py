import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

TARGET_COLUMN = "compressive_strength_mpa"

df = pd.read_csv("student-resources/concrete_compressive_strength.csv")
y = df[TARGET_COLUMN].to_numpy(dtype=np.float64)
x_df = df.drop(columns=[TARGET_COLUMN])

print(x_df.head())
print(df.dtypes)

test_size = 0.15
val_size = 0.15
random_state = 119

x_trainval,x_test,y_trainval,y_test = train_test_split(
    x_df,y,
    test_size=test_size,
    random_state=random_state,
)

val_fraction = val_size/(1.0 - test_size)

x_train,x_val,y_train,y_val = train_test_split(
    x_trainval,y_trainval,
    test_size=val_fraction,
    random_state=random_state,
)
print(f"sizes:{len(x_train),len(x_val),len(x_test)}")