#Import all necessary libraries.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Prepare the data.
data = "PathToFile" #Replace with the path to your file.
df = pd.read_csv(data) #You can replace the csv with excel, json or hdf respective to your file format.

#Scale the data.
scaler = StandardScaler()
X = scaler.fit_transform(df)

#Create the PCA model.
pca = PCA(n_components = 2) #Replace 2 with the number of components you want.

#Fit the PCA model and transform the data.
pca_result = pca.fit_transform(X)

#Print the principal components.
print("PCA :\n", pca_result)

#Print explained variance ratio.
print("Explained Variance Ratio =", pca.explained_variance_ratio_)

#IMPORTANT - This code will show error until the placeholders are replaced.
