import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
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

perceptron = Perceptron(n_iter=40, eta0=0.1, random_state=0)
perceptron.fit(X_train_std, y_train)

y_pred = perceptron.predict(X_test_std)

n_misclassified = sum([1 if y != y_ else 0 for y, y_ in zip(y_test, y_pred)])
print("Misclassified samples: {}".format(n_misclassified))

print("Classification accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))

plot_decision_regions(X=X_train_std, y=y_train, X_test=X_test_std, y_test=y_test, classifier=perceptron)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
