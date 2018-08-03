import numpy as np
import pandas as pd

_UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases"

def load_breast():
    dataname = "breast-cancer-wisconsin"
    target_url = f"{_UCI_BASE}/{dataname}/{dataname}.data"
    mat = pd.read_csv(target_url, header=0, na_values="?").fillna(0)
    X = mat.iloc[:,1:10].values.astype(float)
    y = np.where(mat.iloc[:,10].values == 4, 1, 0)
    return (X, y)

def load_spam():
    dataname = "spambase"
    target_url = f"{_UCI_BASE}/{dataname}/{dataname}.data"
    mat = pd.read_csv(target_url, header=0)
    X = mat.iloc[:,1:].values.astype(float)
    y = mat.iloc[:,0].values.astype(int)
    return (X, y)
