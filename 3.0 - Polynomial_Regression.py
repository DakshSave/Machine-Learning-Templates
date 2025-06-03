#Import all the necessary libraries.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import accuracy_score

#Prepare the data.
data = "_" #Replace the underscore with the file path.
df = pd.read_csv(data) #You can change csv to excel, json or hdf according to your data file.

#Clean the data.
df.dropna(inplace = True)

#Assign the variables.
X = df[["_"]] #Replace the underscore with the header(s) of what would be the independent variable(s).
Y = df["_"] # Replace the underscore with the header of what would be the dependent variable.

#Split the data into training and testing sets.
X_train, X_test, Y_train, Y_test = (X, Y, test_size = _, random_state = 42)

#Scale the data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Transform the data into a polynomial.
poly = PolynomialFeatures(degree = _) #Replace the underscore with what you want to be the degree of the polynomial.
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

#Create a linear regression model.
model = LinearRegression()

#Fit the model with the data.
model.fit(X_train, Y_train)

#Obtain the prediction.
Y_prediction = model.predict(X_test)



#IMPORTANT - This code will show error until the underscores are filled.
