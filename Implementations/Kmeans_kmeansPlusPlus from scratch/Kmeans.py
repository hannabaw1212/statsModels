import numpy as np

from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

class KMeans:
    """
    Implementation of the KMeans clustering algorithm from scratch.

    Attributes:
        n_clusters (int): Number of clusters to form.
        max_iter (int): Maximum number of iterations of the k-means algorithm.
        tol (float): Tolerance for convergence. The algorithm stops if the change in centroids is less than this value.
        init (str): Method for initialization ('kmeans++' for KMeans++ initialization, 'random' for random initialization).
        centroids (np.array): Array of centroids after fitting the model.
        labels (np.array): Labels of each point after fitting the model.

    Methods:
        fit(X): Computes KMeans clustering.
        predict(X): Predicts the closest cluster each sample in X belongs to.
        kmeans_plusplus_init(X, k): Initializes centroids using the KMeans++ algorithm.
    """
    def __init__(self, n_clusters=4, max_iter=300, tol=1e-5, init='kmeans++'):
        """
        Initializes the KMeans instance with specified parameters.

        Parameters:
            n_clusters (int): The number of clusters and centroids to form.
            max_iter (int): Maximum number of iterations for the algorithm.
            tol (float): Tolerance for determining convergence.
            init (str): Method for initializing the centroids ('kmeans++' or 'random').
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """
        Compute k-means clustering.

        Parameters:
            X (np.array): Input data, array of shape (n_samples, n_features).

        Updates:
            centroids (np.array): Coordinates of cluster centers.
            labels (np.array): Index of the cluster each sample belongs to.
        """
        if self.init == "kmeans++":
            self.centroids = self.kmeans_plusplus_init(X, self.n_clusters)

        else:
            self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Compute distances and assign clusters
            ds = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(ds, axis=1)

            # new centroids
            new_centroids = np.array(
                [X[self.labels == i].mean(axis=0) if np.any(self.labels == i) else self.centroids[i] for i in
                 range(self.n_clusters)])

            # Check if we need to stop
            if np.linalg.norm(self.centroids - new_centroids, ord='fro') < self.tol:
                break

            self.centroids = new_centroids

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters:
            X (np.array): New data to predict, array of shape (n_samples, n_features).

        Returns:
            labels (np.array): Index of the cluster for each sample.
        """
        ds = ds = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(ds, axis=1)

    def kmeans_plusplus_init(self, X, k):
        """
        Initialize centroids using the KMeans++ algorithm for better centroid seeding.

        Parameters:
            X (np.array): Input data, array of shape (n_samples, n_features).
            k (int): Number of centroids to initialize.

        Returns:
            centroids (np.array): Initialized centroids.
        """
        n_samples, _ = X.shape
        centroids = np.zeros((k, X.shape[1]))
        # Randomly choose the first centroid from the data points
        centroids[0] = X[np.random.randint(n_samples)]
        for i in range(1, k):
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - centroids[:i], axis=2), axis=1)
            probabilities = distances / distances.sum()
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids[i] = X[j]
                    break
        return centroids


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    kmeans = KMeans(n_clusters=4, max_iter=300, tol=1e-4, init='kmeans++')
    kmeans.fit(X)

    import matplotlib.pyplot as plt

    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=300, marker='*', c='red')  # mark centroids
    plt.title('KMeans Clustering')
    plt.show()
