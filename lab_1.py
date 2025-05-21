from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn import datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pathlib


def least_squares(x, y):
    n = len(x)

    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(xi**2 for xi in x)
    sum_xy = sum(x[i] * y[i] for i in range(n))

    a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    b = (sum_y - a * sum_x) / n

    return a, b


def csv_read():
    filename = input('Введите путь к CSV-файлу: ')
    df = pd.read_csv(pathlib.Path(filename))
    col1_name = df.columns[0]
    col2_name = df.columns[1]
    choice = int(input(f"Выберите оси:\n1) x: {col1_name}, y: {col2_name}\n2) x: {col2_name}, y: {col1_name}\n"))

    if choice == 2:
        X = df[col2_name].values
        Y = df[col1_name].values
        col1_name, col2_name = col2_name, col1_name
    else:
        X = df[col1_name].values
        Y = df[col2_name].values

    print(f"{col1_name}:\ncount: {len(X)}\nmin: {min(X)}\nmax: {max(X)}\navg: {np.mean(X)}")
    print(f"{col2_name}:\ncount: {len(Y)}\nmin: {min(Y)}\nmax: {max(Y)}\navg: {np.mean(Y)}")
    return X, Y


def lab_1_1_out(X, Y):
    plt.figure()
    plt.scatter(X, Y)
    plt.title("График 1")

    a, b = least_squares(X, Y)
    print(f"Параметры регрессионной прямой: {a}, {b}")

    new_Y = [a * i + b for i in X]
    plt.figure()
    plt.scatter(X, Y)
    plt.plot(X, new_Y, color="red")
    plt.title("График 2")

    for x, y, new_y in zip(X, Y, new_Y):
        error = abs(y - new_y)

        plt.gca().add_patch(plt.Rectangle(
            (x - error / 2, y - error / 2),
            width=error,
            height=error,
            edgecolor='black',
            facecolor='none',
            hatch='//',
            alpha=0.3,
            label='Ошибка' if x == X[0] else ""
        ))
    plt.figure()
    plt.scatter(X, Y)
    plt.plot(X, new_Y, color="red")
    plt.title("График 3")

    plt.show()


def lab_1_2(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X.reshape(-1, 1), Y, test_size=0.2, random_state=0)
    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    sklearn_pred = reg.predict(X_test)
    a, b = least_squares(X_train.flatten(), Y_train)
    manual_pred = [a * xi + b for xi in X_test.flatten()]

    print("Scikit-learn: a =", reg.intercept_, ", b =", reg.coef_[0])
    print("Собственный метод: a =", a, ", b =", b)

    results = pd.DataFrame({
        'bmi': X_test.flatten(),
        'target_real': Y_test,
        'pred_sklearn': sklearn_pred,
        'pred_manual': manual_pred
    })

    print(results.head(10))


def lab_1_3(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X.reshape(-1, 1), Y, test_size=0.2, random_state=0)
    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)

    print("Scikit-learn: w0 =", reg.intercept_, ", w1 =", reg.coef_[0])
    print(f"MAE: {mean_absolute_error(Y_test, Y_pred):.2f}")
    print(f"R2: {r2_score(Y_test, Y_pred):.2f}")
    print(f"MAPE: {mean_absolute_percentage_error(Y_test, Y_pred):.2f}")


if __name__ == "__main__":
    """ X, Y = csv_read()
    lab_1_1_out(X, Y)"""
    diabetes = datasets.load_diabetes()
    X, Y = diabetes.data[:, 2], diabetes.target
    lab_1_2(X, Y)
    lab_1_3(X, Y)

