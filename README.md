# Hierarchical-clustering
A Hierarchical clustering method works via grouping data into a tree of clusters. Hierarchical clustering begins by treating every data point as a separate cluster.
# HIERARCHICAL CLUSTERING

# IMORTING THE LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORTING THE DATASET
dataset = pd.read_csv(r"C:\Users\Rafi\Desktop\PROJECTS\Hirarchical clustering\Mall_Customers.csv")
dataset

x = dataset.iloc[:, [3,4]].values
x

# Using the dendogram to find the optimal no.of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()
![download](https://github.com/tayyib1204/Hierarchical-clustering/assets/132560640/f6f53ec6-5a52-4d46-a2d1-2fc0d49d2550)



#Training the hierarchical model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)

# Visualising the clusters
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'cluster1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'cluster2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'cluster3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'cluster4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'cluster5')
plt.title('Clusters of customers')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
![download](https://github.com/tayyib1204/Hierarchical-clustering/assets/132560640/0ed79eb8-326f-4b29-a853-4c4404734180)

