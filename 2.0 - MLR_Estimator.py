#Import all the necessary libraries.
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Pipeline
from sklearn.metrics import mean_squared_error

#Prepare the data.
data = "_" #Replace the underscore with the path to your desired file.
df = pd.read_csv(data) #You can use excel, json or hdf also according to your data.

#Clean the data.
df.dropna(inplace = True)

#Assign the variables.
X = df[["_", "_"]] #Replace the underscore(s) with the header(s) of the column(s) of the independent variable(s).
Y = df["_"] #Replace the underscore with the header of the column of the dependent/target variable.

#Split the data into training and testing sets.
X_train, X_test, Y_train, Y_test = (X, Y, test_size = _, random_state = 42) #The value of test_size (where the underscore is) must be between 0 and 1 (representing percentages). 

#Create a pipeline.
pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])

#Fit the pipeline with the data.
pipeline.fit(X_train, Y_train)

#Make separate variables of each step for better readability (optional)
model = pipeline.named_steps["model"]
scaler = pipeline.named_steps["scaler"]

#Get the prediction.
Y_prediction = pipeline.predict(X_test)

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
