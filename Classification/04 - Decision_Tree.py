#Import all necessary libraries.
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#Prepare the data.
data = "_" #Replace the underscore with the path to you file.
df = pd.read_csv(data) #You can replace the csv with excel, json or hdf respective to your data file.

#Clean the data.
df.dropna(inplace = True)

#Assign the variables.
X = df[["_"]] #Replace the underscore(s) with the header(s) of the independent variable(s).
Y = df["_"] #Replace the underscore with the header of the independent variable.

#Split the data into training and testing sets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = _, random_state = 42) #The value of test size must be between 0 and 1 (representating percentages).

#Create a pipeline.
pipeline = Pipeline([("scaler", StandardScaler()), ("model", DecisionTreeClassifier(max_depth = _, min_samples_split = _, min_samples_split = _, min_samples_leaf = _, max_leaf_nodes = _, random_state = 42))])
'''
IMPORTANT
Hyperparameters:
max_depth = Maximum depth of the tree.
min_samples_split = Minimum number of samples required to split a node.
min_samples_leaf = Minimum number of samples required to be at a leaf node.
max_leaf_nodes = Maximum number of leaf nodes.
max_features = Maximum number of features considered for a split.
'''

#Make separate variables of each step for better readability (optional).
scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["model"]

#Fit the pipeline with the training data.
pipeline.fit(X_train, Y_train)

#Obtain the predictions.
Y_prediction = pipeline.predict(X_test)

#Get the accuracy score.
accuracy = accuracy_score(Y_test, Y_prediction)

#Print the predictions and the accuracy score.
print("Predictions :\n", Y_prediction)
print("Accuracy =", accuracy)

#IMPORTANT - This code will show error until the underscores are filled.
