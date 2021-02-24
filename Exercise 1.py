import numpy as np 

X = np.array([
    [1,2],
    [0,0],
    [-1, -2],
    [-2, -4],
    [2,4]
    ])

'''__________________________________________________________________________________________________________________'''
''' a)  Standardise the dataset to form a matrix '''

print(f'The Matrix X is:\n {X}\n')
column_means = X.mean(0)
print(f'The column means are: {np.round(column_means,2)}')

column_stds = X.std(0)
print(f'The column standard deviations are: {np.round(column_stds,2)}\n')

eps = 1e-8 # a small constant to prevent dividing by zero
X_s = (X - column_means)/(column_stds + eps)
X_s = np.round(X_s,2)

column_means = X_s.mean(0)
print(f'The column means are: {np.round(column_means,2)}')
column_stds = X_s.std(0)
print(f'The column standard deviations are: {np.round(column_stds,2)}\n')

print(f'The final standardised Matrix X_s is: \n{X_s}\n')

'''__________________________________________________________________________________________________________________'''
''' b) Computing the principal components of X_s. They should be in unit norm'''

print(f'The X_s function has a shape of: {X_s.shape}\n')

D = 2
C = (1/D) * X_s.T @ X_s  # Constructing the covariance matrix C = 1/D * X.T * X  <- X normalised

eigenvalues, eigenvectors = np.linalg.eig(C)

print(f'The eigenvalues of C are: {eigenvalues}')
print(f'The eigenvectors of C are: \n{eigenvectors}\n')

'''__________________________________________________________________________________________________________________'''
''' c) Use the principal components to transform your standardised dataset down to 1 dimension'''

W = eigenvectors[:,0:1] # Get the first principal component
print(f'{W}\n')

X_reduced = X_s @ W     # The matrix reduced to one dimension
X_tilde = X_s @ W @ W.T # reconstructed X_s after reduction

print(f'The reduced version to 1 dimension of the dataset is: \n {X_reduced}\n')
print(f'The reconstructed X_s Matrix after 1 dimension is: {X_tilde}')

E = np.sum(np.sum((X_s-X_tilde)**2, 1)**0.5)
print(f'The final error of the reconstruction is equal to: {E}')
print(f'The rounded error of the reconstruction is equal to: {np.round(E,2)}')




