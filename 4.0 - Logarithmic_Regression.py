#Import the necessary libraries.
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

#Prepare the data.
data = "_" #Replace the underscore with the path to your file.
df = pd.read_csv(data) #If the data is in another format replace the csv with your file format (csv, excel, hdf, json).

#Clean the data.
df.dropna(inplace = True)

#Assign the variables.
X = np.log(df[["_"]]) #Replace the underscore with the header(s) of the independent variable(s).
Y = df["_"] #Replace the underscore with the header of the dependent variable.

#Split the data into training and testing sets.
X_train, X_test, Y_train, Y_test = (X, Y, test_size = _, random_state = 42) #The value of test size must be between 0 and 1 (which represent precentages).

#Scale the data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Create a linear regression model.
model = LinearRegression()

#Fit the model with the variables.
model.fit(X_train, Y_train)

#Obtain the predictions.
Y_prediction = model.predict(X_test)

#Get the MSE.
mse = mean_squared_error(Y_test, Y_prediction)

#Print the predictions and the MSE.
print("Predictions :\n", Y_prediction)
print("Mean Squared Error =", mse)

#IMPORTANT - This code will show error until the underscores are filled.
