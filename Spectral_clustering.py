import numpy as np

class SpectralClustering:

    def __init__(self,k):
        self.k = k

    def pairwise_distances(self,X, Y):

        distances = np.empty((X.shape[0], Y.shape[0]), dtype='float')

        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                distances[i, j] = np.linalg.norm(X[i]-Y[j])

        return distances

    def nearest_neighbor_graph(self,X):
        
        X = np.array(X)

        n_neighbors = min(int(np.sqrt(X.shape[0])), 10)

        A = pairwise_distances(X, X)

        sorted_rows_ix_by_dist = np.argsort(A, axis=1)

        nearest_neighbor_index = sorted_rows_ix_by_dist[:, 1:n_neighbors+1]

        W = np.zeros(A.shape)

        for row in range(W.shape[0]):
            W[row, nearest_neighbor_index[row]] = 1

        for r in range(W.shape[0]):
            for c in range(W.shape[0]):
                if(W[r,c] == 1):
                    W[c,r] = 1

        return W

    def compute_laplacian(self,W):
        
        d = W.sum(axis=1)

        #create degree matrix
        D = np.diag(d)

        # Laplacian Matrix equal to degree matrix-Adjucancy matrix
        L =  D - W
        return L

    def get_eigvecs(self,L):
        
        eigvals, eigvecs = np.linalg.eig(L)
        
        ix_sorted_eig = np.argsort(eigvals)[:self.k]

        return eigvecs[:,ix_sorted_eig]

    def fit_predict(self,X):

        #create weighted adjacency matrix
        W = nearest_neighbor_graph(X)

        #create unnormalized graph Laplacian matrix
        L = compute_laplacian(W)

        #create projection matrix with first k eigenvectors of L
        E = get_eigvecs(L)

        #return clusters using k-means on rows of projection matrix
        labels = k_means_clustering(E, self.k)

        return np.ndarray.tolist(labels)