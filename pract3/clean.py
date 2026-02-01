# import pandas as pd
import numpy as np
import pandas as pd



CSV_PATH = "./data/ames_curated.csv"

SEED = 4320
TARGET_COL = "saleprice"

POSSIBLE_EXCLUDES = ["pid", TARGET_COL]
# seed repeatable equal randommness n is our number of items
def split_indices(n:int, seed:int, train_frac: float =0.70 , val_frac: float = 0.15):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    # perm is an array of randomly sorted int from 0 --> n-1
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train+n_val]
    testing_idx = perm[n_train+n_val:]
    print(f"training idx: \n {train_idx}, \n validation idx: \n {val_idx},  \n testings idx: \n {testing_idx}")
    return train_idx, val_idx, testing_idx

def main ():
    df = pd.read_csv("./data/ames_curated.csv")

    train_idx, val_idx,test_idx = split_indices(len(df),SEED)

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()
    print(f"data frames trainingDF: \n {train_df},\n Validation DF: \n{val_df},\n TEST DF:\n{test_df}")

    # seperate what we want to target
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    y_val = val_df[TARGET_COL].to_numpy(dtype=float)
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    print(f"our target columns \n traiining targets :\n {y_train},\n validation targets: \n{y_val},\n testing targets :\n {y_test}")

    # we want to remove our targets and anything not necassary to train model

    x_train = train_df.drop(columns = [c for c in POSSIBLE_EXCLUDES if c in train_df.columns])
    x_val = val_df.drop(columns = [c for c in POSSIBLE_EXCLUDES if c in val_df.columns])
    x_test = test_df.drop(columns= [c for c in POSSIBLE_EXCLUDES if c in test_df.columns])

    numeric_columns = [c for c in x_train.columns if pd.api.types.is_numeric_dtype(x_train[c])]
    cat_columns = [c for c in x_train.columns if c not in numeric_columns]

    # NUMERIC selction of training data
    x_train_numeric =x_train[numeric_columns]

    # series of medians 
    x_train_medians = x_train_numeric.median()
    # categorial selection of training data
    x_train_cat = x_train[cat_columns]
    # series of modes
    x_train_cat_mode = x_train_cat.mode()

    # look at na's

    print(x_train_medians)

    x_imputed = x_train_numeric.fillna(x_train_medians)

    print(x_imputed.isnull())

main()