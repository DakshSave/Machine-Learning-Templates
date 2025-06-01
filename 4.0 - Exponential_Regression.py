#Import the necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Prepare the data.
data = "_" #Replace the underscore with the path to the file.
df = pd.read_csv(data) #If the data is in another format replace the csv with your file format (csv, excel, hdf, json).

#Clean the data.
df.dropna()

#Assign the variables.
X = df["_"] #Replace the underscore with the header of the independent variable.
Y = df["_"] #Replace the underscore with the header of the dependent variable.

#Create the exponential function.
def exponential_func(x, a, b):
    return a*np.exp(b*x)

#Fit the function and obtain the parameters.
params, _ = curve_fit(exponential_func, X, Y)
a, b = params

#Obtain the predictions.
Y_prediction = exponential_func(X, a, b)

#Plot the actual data and the predicted data.
plt.scatter(X, Y, color="blue", label="Actual Data")
plt.plot(X, Y_prediction, color="red", label="Predicted Data")
plt.xlabel("X")
plt.ylabel("Y / Y_prediction")
plt.title("Exponential Regression")
plt.legend()
plt.grid(True)
plt.show()

#IMPORTANT - This code will show error until the underscores are filled.
