#Import all necessary libraries.
import pandas as pd
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler

#Prepare the data.
data = "_" #Replace the underscore with the path to your file.
df = pd.read_csv(data) #You can replace the csv with excel, json or hdf respective to your file format.

#Clean the data.
df.dropna(inplace = True)

#Assign the variables.
X = df[["_"]] #Replace the underscore(s) with header(s) of your independent variable(s).

#Scale the data.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Create the model.
model = hdbscan.HDBSCAN(min_cluster_size = _, min_samples = _) #Replace the underscores with desired values.

#Fit the model with data and obtain clusters.
clusters = model.fit_predict(X_scaled)

#Add cluster labels to the dataframe.
df["Cluster"] = clusters

#Print the cluster assignments.
print("Cluster assignments :\n", clusters)

#IMPORTANT - This code will show error until the underscores are filled.
