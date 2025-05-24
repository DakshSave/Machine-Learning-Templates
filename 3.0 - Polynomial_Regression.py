#Import all the necessary libraries.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Prepare the data.
data = "_" #Replace the underscore with the file path.
df = pd.read_csv(data) #Change the csv to any other format respective to your file.

#Clean the data.
df.dropna()

#Assign the variables.
X = df[["_"]] #Replace the underscore with the header(s) of what would be the independent variable(s).
Y = df["_"] # Replace the underscore with the header of what would be the dependent variable.

#Transform the data into a polynomial.
poly = PolynomialFeatures(degree = _) #Replace the underscore with what you want to be the degree of the polynomial.
X_poly = poly.fit_transform(X)

#Create a linear regression model.
pr_model = LinearRegression()

#Fit the model with the data.
pr_model.fit(X_poly, Y)

#Obtain the prediction.
Y_prediction = pr_model.predict(X_poly)

#Plot the actual and predicted data.
