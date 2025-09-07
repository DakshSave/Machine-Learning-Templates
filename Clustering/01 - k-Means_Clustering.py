#Import all necessary libraries.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Prepare the data.
data = "_" #Replace the underscore with the path to your file.
df = pd.read_csv(data) #You can replace the csv with excel, json or hdf respective to your file format.

#Clean the data.
df.dropna(inplace = True)

#Assign the variables (only independent variables, since clustering is unsupervised).
X = df[["_"]] #Replace the underscore(s) with header(s) of your independent variable(s).

#Create a pipeline.
pipeline = Pipeline([("scaler", StandardScaler()), ("model", KMeans(n_clusters = _, random_state = 42))]) #Replace the underscore with number of clusters.

#Create separate variables of each step in the pipeline for better readability.
scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["model"]

#Fit the model with data.
pipeline.fit(X)

#Obtain the cluster predictions.
clusters = pipeline.predict(X)

#Add cluster labels to the dataframe.
df["Cluster"] = clusters

#Print the assigned clusters.
print("Cluster assignments :\n", clusters)

#IMPORTANT - This code will show error until the underscores are filled.
