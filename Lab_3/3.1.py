from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)

    print(f"prediction: {y_pred}")
    print(f"actual: {y_test}")
    print(f"accuracy: {score}")

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    #task1
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    colors = ['red', 'green', 'blue']

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i in range(3):
            plt.scatter(X[i*50:i*50+50:1, 0], X[i*50:i*50+50:1, 1], color=colors[i], label=target_names[i])
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title(f'Sepal length - Sepal width')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(3):
        plt.scatter(X[i*50:i*50+50:1, 2], X[i*50:i*50+50:1, 3], color=colors[i], label=target_names[i])
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.title('Petal length - Petal width')
    plt.legend()
    plt.show()

    #task2
    iris_df = pd.DataFrame(X, columns=iris.feature_names)
    iris_df['data'] = [target_names[i] for i in y]

    sns.pairplot(iris_df, hue='data')
    plt.show()

    #task3
    X_setosa_Versicolor = X[0:100]
    y_setosa_Versicolor = y[0:100]

    X_Versicolor_Virginica = X[50:150]
    y_Versicolor_Virginica = y[50:150]

    #task4-8
    print("Setosa - Versicolor")
    train(X_setosa_Versicolor, y_setosa_Versicolor)
    print("Versicolor - Virginica")
    train(X_Versicolor_Virginica, y_Versicolor_Virginica)

    #task9
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,random_state=1, n_clusters_per_class=1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    plt.title("rand data")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.show()

    print("rand data")
    train(X, y)