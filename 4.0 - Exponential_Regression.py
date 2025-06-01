#Import the necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


data = "_"


df = pd.read_csv(data)


X = df["_"]
Y = df["_"]


def exponential_func(x, a, b):
    return a*np.exp(b*x)


params, _ = curve_fit(exponential_func, X, Y)
a, b = params


Y_prediction = exponential_func(X, a, b)

#Plot the data.
plt.scatter(X, Y, color="blue", label="Actual Data")
plt.plot(X, Y_prediction, color="red", label="Predicted Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Exponential Regression")
plt.legend()
plt.grid(True)
plt.show()
