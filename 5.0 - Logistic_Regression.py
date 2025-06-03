#Import the necessary libraries.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Prepare the data.
data = "_" #Replace the underscore with the path to your file.
df = pd.read_csv(data) #If the data is in another format replace the csv with your file format (csv, excel, hdf, json).

#Clean the data.
X = df[["_"]].values #Replace the underscore with the header of the independent variable.
Y = df["_"] #Replace the underscore with the header of the dependent variable.

#Split the data into training and testing sets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = _, random_state = 42) #The value of test size must be between 0 and 1 (which represent precentages).

#Scale the data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Create the logistic regression model.
model = LogisticRegression()

#Fit the model
model.fit(X_train, Y_train)

#Create a range of values for prediction.
X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

#Scale the range
X_range = scaler.transform(X_range)

#Obtain the predictions.
Y_prediction = model.predict(X_test)

#Obtain predicted probabilities.
Y_probability = model.predict_proba(X_range)[:, 1]

#Get the accuracy score.
accuracy = accuracy_score(Y_test, Y_prediction)

#Print the probability and accuracy score.
print("Probability =", Y_probability)
print("Accuracy =", accuracy)
