import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

TARGET_COLUMN = "compressive_strength_mpa"

df = pd.read_csv("student-resources/concrete_compressive_strength.csv")
y = df[TARGET_COLUMN].to_numpy(dtype=np.float64)
x_df = df.drop(columns=[TARGET_COLUMN])
print(y)
print(len(x_df.columns))
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

numeric_col = [c for c in x_df.columns if is_numeric_dtype(x_df[c])]
cat_col = [c for c in x_df.columns if c not in numeric_col]

print(f"numeric columns: \n {numeric_col}")
print(f"categorial columns: \n {cat_col}")

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

pre = ColumnTransformer(
    transformers=[
        ("num", num_pipe, numeric_col),
        ("cat", cat_pipe, cat_col),
    ],
    remainder="drop"
)
print(x_train)
x_train_p = pre.fit_transform(x_train)
x_val_p = pre.transform(x_val)
x_test_p = pre.transform(x_test)

print(f"Xtrain_p shape: \n {x_train_p}")
x_train_p = np.asarray(x_train_p, dtype=np.float64)
x_val_p = np.asarray(x_val_p,dtype=np.float64)
x_test_p = np.asarray(x_test_p,dtype=np.float64)

print(f"Xtrain_p shape: \n {x_train_p.shape}")

def add_bias_column(x):
    new_column = np.ones((x.shape[0],1))
    xb = np.hstack((new_column,x))

    # print(xb.shape)

    # print(xb[:,0])
    assert xb.shape == (x.shape[0],x.shape[1] + 1)
    return xb
    
    
def predict(x,w):
    xb = add_bias_column(x_train)
    y_hat = x @ w
    return y_hat

# y_hat = predict(x_train_p,)

def mse_loss(xb,y,w):
    y_hat = xb @ w
    return np.mean((y_hat-y)**2)

def mse_grad(xb,y,w):
    error = (xb @ w)-y
    weights = xb.T

    return (2/len(y)) * (weights @ error)

xb_tr = add_bias_column(x_train_p)
xb_val = add_bias_column(x_val_p)



train_losses = []
val_losses = []
epochs = 100
lr = 0.1
def main():
    w = np.zeros(xb_tr.shape[1],dtype=np.float64)
    for epoch in range(epochs):
        grad = mse_grad(xb_tr,y_train,w)
        w = w-lr * grad
        
        train_losses.append(mse_loss(xb_tr,y_train,w))
        val_losses.append(mse_loss(xb_val,y_val,w))

        if (epoch+1) % 100 == 0:print(epoch+1,train_losses[-1],val_losses[-1])

main()


