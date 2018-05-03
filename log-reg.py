import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_access import load_iris_data
from plots.learning_results import plot_decision_regions

iris_df = load_iris_data()

predictor_columns = ["petal length (cm)", "petal width (cm)"]
target_column = "int class"

X = iris_df[predictor_columns].values
y = iris_df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
plot_decision_regions(X=X_train_std, y=y_train, X_test=X_test_std, y_test=y_test, classifier=lr)

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
