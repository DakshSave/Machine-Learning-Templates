#Import all the necessary libraries.
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

#Prepare the data.
data = "_" #Replace the underscore with the path to your desired file.
df = pd.read_csv(data) #You can use excel, json or hdf also according to your data.

#Clean the data.
df.dropna(inplace = True)

#Assign the variables.
X = df[["_"]] #Replace the underscore with the header of the column of the independent variable.
Y = df["_"] #Replace the underscore with the header of the column of the dependent/target variable.

#Split the data into training and testing sets.
X_train, X_test, Y_train, Y_test = (X, Y, test_size = _, random_state = 42) #The value of test_size (where the underscore is) must be between 1 and 0 (representing percentages). 

#Scale the data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Create the linear regression model.
model = LinearRegression()

#Fit the model with the data.
model.fit(X_train, Y_train)

#Get the prediction.
Y_prediction = model.predict(X_test)

#Get the slope and the intercept.
slope = model.coef_
intercept = model.intercept_

#Get the accuracy score.
mse = mean_squared_error(Y_test, Y_prediction)

#Print the slope, intercept, prediction and the accuracy score.
print("Slope =", slope)
print("Intercept =", intercept)
print("Predictions :\n", Y_prediction)
print("Mean Squared Error =", mse)

#IMPORTANT - This code will show error until the underscores are filled.
