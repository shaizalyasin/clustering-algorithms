# Clustering Algorithms: K-means and DBSCAN  

This repository contains the implementation of two widely used clustering algorithms: **K-means** and **DBSCAN**. The project focuses on building a robust and functional framework for **unsupervised learning**, enabling the discovery of natural groupings within a dataset.  

## 📊 Core Algorithms and Features  

### K-means  
A classic partitioning method to divide a dataset into a predefined number of clusters. Key features:  

- **Initialization**: Random selection of initial cluster centroids.  
- **Assignment**: Assigning each data point to the nearest centroid.  
- **Centroid Update**: Recalculating cluster centroids.  
- **Iterative Process**: Runs until assignments stabilize, ensuring convergence.  

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)  
A density-based algorithm capable of discovering arbitrarily shaped clusters and identifying outliers. Key features:  

- **Point Classification**: Distinguishes core, border, and noise points.  
- **Neighborhood Search**: Finds all points within a specified radius (*epsilon*).  
- **Cluster Expansion**: Recursively grows clusters from core points.  
- **Noise Handling**: Identifies and labels non-clustered points as noise.  

---
