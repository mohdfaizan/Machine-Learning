import numpy as np
import pandas as pd

np.random.seed(42)

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit_predict(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # covariance, function needs samples as columns
        cov = np.cov(X.T)
        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

        return np.dot(X, self.components.T)

    def reverse_transform(self,X_tra):
        revpca = np.dot(X_tra[:,:self.n_components], self.components[:self.n_components,:])
        revpca += self.mean
        return revpca

    def re_error(self,data,re_data):
        error = data - re_data
        error_var = 0
        for i in range(data.shape[1]):
            error_var += np.linalg.norm(error[:,i])**2
        error_var = error_var/data.shape[1]
        return error_var


if __name__=='__main__':
    
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = pd.DataFrame(iris.data)
    data['original_label'] = iris['target']
    data.loc[data['original_label']==0,'original_label']=3

    pca = PCA(2)
    predicted_data = pca.fit_predict(data.iloc[:,:-1].values)
    reversed_data = pca.reverse_transform(predicted_data)
    error_varience = pca.re_error(data.iloc[:,:-1].values,reversed_data)
    