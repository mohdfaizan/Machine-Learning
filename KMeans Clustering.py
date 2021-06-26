import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
np.random.seed(42)

# Defining K-Means Clustering
class KMeans():
    
    def euclidean_distance(self,x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def __init__(self, K=2, max_iters=100):
        self.K = K
        self.max_iters = max_iters
        
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def fit_predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            
            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break
            
        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx+1
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        
        # Run a loop for all the data points available in dataset
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [self.euclidean_distance(sample, point) for point in centroids]
        
        # Store the closest centroid index
        closest_index = np.argmin(distances)
        
        return closest_index

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, for all centroids
        distances = [self.euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0


if __name__=='__main__':

    # path = 'http://cs.joensuu.fi/sipu/datasets/jain.txt'
    path = sys.argv[1]
    
    data = pd.read_csv(path, delimiter = "\t",header=None)
    kmeans = KMeans(K=2)
    # fit_predict method of kmeans will take  numpy array as input
    label = kmeans.fit_predict(data[[0,1]].values)
    data['outlabel'] = label.astype(np.int64)
    correct = 0
    for i in range(data.shape[0]):
        if label[i]==data.iloc[i,2]:
            correct+=1
    incorrect = data.shape[0]-correct
    print("Correct labels: ",correct," Incorrect labels: ",incorrect)
    print("Percentage of Correct classification : ",round((correct*100)/data.shape[0],2),"%")
    plt.figure()
    plt.title('K-Means Clustering')
    plt.scatter(data[0],data[1],c=data['outlabel'])
    plt.show()