#Import all necessary libraries.
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Prepare the data.
data = "_" #Replace the underscore with the path to your file.
df = pd.read_csv(data) #You can replace the csv with excel, json or hdf respective to your file format.

#Clean the data.
df.dropna(inplace = True)

#Assign the variables (independent variables only).
X = df[["_"]] #Replace the underscore(s) with header(s) of your independent variable(s).

#Create a pipeline.
pipeline = Pipeline([("scaler", StandardScaler()), ("model", AgglomerativeClustering(n_clusters = _, linkage = "_"))]) #Replace the underscore with number of clusters. Linkage options: "ward", "complete", "average", "single".

#Create separate variables of each step in the pipeline for better readability.
scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["model"]

#Fit the model with data.
X_scaled = scaler.fit_transform(X)   # AgglomerativeClustering does not support pipeline .fit(X)
clusters = model.fit_predict(X_scaled)

#Add cluster labels to the dataframe.
df["Cluster"] = clusters

#Print the assigned clusters.
print("Cluster assignments :\n", clusters)

#IMPORTANT - This code will show error until the underscores are filled.
