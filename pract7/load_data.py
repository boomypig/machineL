from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier

# --------------------------------------------------
# 5. Load fixed splits
# --------------------------------------------------
def createSplits(matrices, messages):

    y = messages["label"].to_numpy(dtype=int)

    print("Label distribution:", np.bincount(y))


    with open("./data/split.json", "r") as f:

        split = json.load(f)

    train_ids = np.array(split["train_ids"], dtype=int)
    val_ids   = np.array(split["val_ids"], dtype=int)
    test_ids  = np.array(split["test_ids"], dtype=int)

    print("Train/Val/Test sizes:",
        len(train_ids), len(val_ids), len(test_ids))
    # --------------------------------------------------
    # 6. Create split datasets
    # --------------------------------------------------

    # Choose which representation to use:
    X = matrices       # for MultinomialNB or kNN (recommended)     

    X_train = X[train_ids]
    X_val   = X[val_ids]
    X_test  = X[test_ids]

    y_train = y[train_ids]
    y_val   = y[val_ids]
    y_test  = y[test_ids]
    print("Training feature matrix shape:", X_train.shape)

    return X_train,X_val,X_test,y_train,y_val,y_test



def main():
    DATA_DIR = Path("./data")  # adjust if needed

    # --------------------------------------------------
    # 2. Load raw messages (for error analysis)
    # --------------------------------------------------

    messages = pd.read_csv(DATA_DIR / "messages.csv")

    print("Messages shape:", messages.shape)
    print(messages.head())
    print(messages.columns)

    # --------------------------------------------------
    # 3. Load feature matrices (sparse!)
    # --------------------------------------------------

    X_counts = sparse.load_npz(DATA_DIR / "X_counts.npz")
    X_tfidf  = sparse.load_npz(DATA_DIR / "X_tfidf.npz")

    print("Counts matrix shape:", X_counts.shape)
    print("TF-IDF matrix shape:", X_tfidf.shape)
    # --------------------------------------------------
    # 4. Load labels
    # --------------------------------------------------
   
    x_train_multi,x_val_multi,x_test_multi,y_train_multi,y_val_multi,y_test_multi = createSplits(X_counts,messages) # for MultinomialNB

    x_train_knn,x_val_knn,x_test_knn,y_train_knn,y_val_knn,y_test_knn = createSplits(X_tfidf,messages)  # for kNN

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(x_train_knn,y_train_knn)

main()

    
    