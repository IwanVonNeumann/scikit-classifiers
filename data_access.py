import numpy as np
import pandas as pd

from sklearn import datasets


def load_iris_data():
    iris = datasets.load_iris()
    n = iris.target.shape[0]
    merged_data = np.append(iris.data, iris.target.reshape((n, 1)), axis=1)
    column_names = iris.feature_names[:]
    column_names.append("int class")
    df = pd.DataFrame(data=merged_data, columns=column_names)
    df[["int class"]] = df[["int class"]].astype(int)
    return df
