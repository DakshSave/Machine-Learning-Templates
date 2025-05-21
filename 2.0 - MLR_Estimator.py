#Import all the necessary libraries.
import pandas as pd
from sklearn.linear_model import LinearRegression

#Prepare the data.
data = "_" #Replace the underscore with the path to your desired file.
df = pd.read_csv(data) #You can use excel, json or hdf also according to your data.

#Clean the data.
df = df.dropna()

#Assign the variables.
X = df[["_", "_"]] #Replace the underscores with the headers of the columns of the independent variables. Adjust the number according to the use.
Y = df["_"] #Replace the underscore with the header of the column of the dependent/target variable.

#Create the linear regression model.
lr_model = LinearRegression()

#Fit the model with the data.
lr_model.fit(X, Y)

#Get the prediction.
Y_prediction = lr_model.predict(X)

#Get the coefficients and the intercept.
coefficients = lr_model.coef_
intercept = lr_model.intercept_

#Print the coefficients, intercept and the prediction.
print("Coefficients =", coefficients)
print("Intercept =", intercept)
print("Predictions :")
for prediction in Y_prediction:
    print(prediction)

#IMPORTANT - This code will show error until the underscores are filled.
