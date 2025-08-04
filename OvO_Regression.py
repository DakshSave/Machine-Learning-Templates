#Import all necessary libraries.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsOneRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

#Prepare the data.
data = "_" #Replace the underscore with the path to your file.
df = pd.read_csv(data) #You can replace the csv with excel, json or hdf respective to your file format.

#Clean the data.
df.dropna(inplace = True)

#Assign the variables.
X = df[["_"]] #Replace the underscore(s) with header(s) of your independent variable(s).
Y = df["_"] #Replace the underscore with the header of your dependent variable.

#Split the data into training and testing sets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = _, random_state = 42) #The value of test size must be between 0 and 1 (represents percentages).

#Create a pipeline.
pipeline = Pipeline([("scaler", StandardScaler()), ("model", OneVsOneRegressor(LinearRegression()))])

#Create separate variables of each step in the pipeline for better readability.
scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["model"]

#Fit the model with training data.
pipeline.fit(X_train, Y_train)

#Obtain the predictions.
Y_prediction = pipeline.predict(X_test)

#Get the accuracy score.
mse = mean_squared_error(Y_test, Y_prediction)

#Print the predictions and accuracy score.
print("Predictions :\n", Y_prediction)
print("MSE =", mse)

#IMPORTANT - This code will show error until the underscores are filled.
