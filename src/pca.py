import numpy as np
from scipy import linalg as LA

def pca(X, k):
    print('**************')

    # centering the data matrix
    mu = np.mean(X, axis = 0)
    print('mean vector: ' + str(mu))
    X_centered = X - mu  
    print('centered data:\n ' + str(X_centered))

    Sigma = np.cov(X_centered, rowvar = False)

    print('***** SVD decomposition ****')

    U, S, V = LA.svd(Sigma)
    print('U:\n' + str(U))
    print('S:\n' + str(S))
    print('V:\n' + str(V))

    #Verifying that eivenvector are indeed unit vectors
    print('Norms of eigenvectors (columns of U):')
    print(np.linalg.norm(U[:,0]), np.linalg.norm(U[:,1]), np.linalg.norm(U[:,2]))

    print('Percentages of variation:\n', str(S/np.sum(S) * 100))

    # If we wish to reduce dimensionality to k, we first need to compute W
    W = U[:, 0:k]
    print('U_redux:\n', str(W))

    #Now we compute Z, the matrix of projected points in k-dimensional space.
    Z = np.matmul(W.T,X.T)
    print('Projected dataset (k=' + str(k) + '):\n' + str(Z.T))
    
    return W, Z

def reconstruct_data(W, Z):
    print('Reconstructed dataset:\n' + str(np.matmul(W, Z).T))