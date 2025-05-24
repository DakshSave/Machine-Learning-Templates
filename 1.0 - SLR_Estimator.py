#Import all the necessary libraries.
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Prepare the data.
data = "_" #Replace the underscore with the path to your desired file.
df = pd.read_csv(data) #You can use excel, json or hdf also according to your data.

#Clean the data.
df = df.dropna()

#Assign the variables.
X = df[["_"]] #Replace the underscore with the header of the column of the independent variable.
Y = df["_"] #Replace the underscore with the header of the column of the dependent/target variable.

#Create the linear regression model.
lr_model = LinearRegression()

#Fit the model with the data.
lr_model.fit(X, Y)

#Get the prediction.
Y_prediction = lr_model.predict(X)

#Get the slope and the intercept.
slope = lr_model.coef_
intercept = lr_model.intercept_

#Print the slope, intercept and the prediction.
print("Slope =", slope)
print("Intercept =", intercept)
print("Predictions :\n", Y_prediction)

#Plot the variables and the prediction.
plt.scatter(X, Y, color = "blue", label = "Actual Data")
plt.plot(X, Y_prediction, color = "red", label = "Predicted Data")
plt.xlabel("X")
plt.ylabel("Y / Y_prediction")
plt.title("Simple Linear Regression Estimator")
plt.legend()
plt.grid(True)
plt.show

#IMPORTANT - This code will show error until the underscores are filled.
