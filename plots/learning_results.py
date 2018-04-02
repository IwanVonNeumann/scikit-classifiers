import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, X_test=None, y_test=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('v', 'o', '^', 's', 'x')
    colors = ('blue', 'red', 'lime', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    b = 0.5
    x1_min, x1_max = X[:, 0].min() - b, X[:, 0].max() + b
    x2_min, x2_max = X[:, 1].min() - b, X[:, 1].max() + b
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

        if X_test is not None and y_test is not None:
            plt.scatter(x=X_test[y_test == cl, 0], y=X_test[y_test == cl, 1], alpha=1, c=cmap(idx), s=60,
                        marker=markers[idx], edgecolors="black")
