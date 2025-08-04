#Import all the necessary libraries.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, Pipeline
from sklearn.metrics import mean_squared_error

#Prepare the data.
data = "_" #Replace the underscore with the file path.
df = pd.read_csv(data) #You can change csv to excel, json or hdf according to your data file.

#Clean the data.
df.dropna(inplace = True)

#Assign the variables.
X = df[["_"]] #Replace the underscore with the header(s) of what would be the independent variable(s).
Y = df["_"] # Replace the underscore with the header of what would be the dependent variable.

#Split the data into training and testing sets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = _, random_state = 42)

#Create a pipeline
pipeline = Pipeline([("scaler", StandardScaler()), ("poly", PolynomialFeatures(degree = _)), ("model", LinearRegression())]) #Replace the underscore with what you want to be the degree of the polynomial.

#Fit the pipeline with the data.
pipeline.fit(X_train, Y_train)

#Create separate variables of each step for better readability (optional).
scaler = pipeline.named_steps["scaler"]
poly = pipeline.named_steps["poly"]
model = pipeline.named_steps["model"]

#Obtain the prediction.
Y_prediction = pipeline.predict(X_test)

#Get the mean squared error.
mse = mean_squared_error(Y_test, Y_prediction)

#Print the predictions and mean squared error.
print("Predictions :\n", Y_prediction)
print("Mean Squared Error =", mse)

#IMPORTANT - This code will show error until the underscores are filled.
