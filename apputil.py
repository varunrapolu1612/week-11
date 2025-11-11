import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from time import time
from statistics import mean

#loading diamonds dataset
diamonds = sns.load_dataset('diamonds')

#Identify numerical columns and save dataframe
numerical_cols = diamonds.select_dtypes(include=[np.number])
NUMERIC_DIAMONDS = numerical_cols.copy()

def kmeans(X, k):
    """performs k-means clustering on a numerical NumPy array `X`

    Parameters:
        X (np.ndarray): A numerical NumPy array of shape (n_samples, n_features).
        k (int): The number of clusters to form.

    Returns:
        centroids (np.ndarray): A NumPy array of shape (k, n_features) representing the cluster centroids.
        labels (np.ndarray): A NumPy array of shape (n_samples,) containing the cluster label for each sample.
    """
    model = KMeans(n_clusters=k, random_state=42, n_init='auto')
    model.fit(X)
    centroids = model.cluster_centers_
    labels = model.labels_
    return centroids, labels

def kmeans_diamonds(n, k):
    """
    Runs k-means clustering on the first `n` rows of the NUMERIC_DIAMONDS dataset.

    Parameters:
        n (int): The number of rows from the NUMERIC_DIAMONDS dataset to use
        k (int): The number of clusters to form.

    Returns:
        centroids (np.ndarray): A NumPy array of shape (k, n_features) representing the cluster centroids.
        labels (np.ndarray): A NumPy array of shape (n_samples,) containing the cluster label for each sample.
    """
    subset = NUMERIC_DIAMONDS.iloc[:n].to_numpy()
    return kmeans(subset, k)

def kmeans_timer(n, k, n_iter=5):
    total_times = 0

    for _ in range(n_iter):
        start_time = time()
        kmeans_diamonds(n, k)
        t = time() - start_time
        total_times += t

    return total_times / n_iter
