import numpy as np

# num1 = input('Enter dimension of matrix n (n*n): ')
#n = input("Enter value: ")

n = int(input("Enter your value: "))
# create a random matrix
A = np.random.randint(-100,100,size=(n, n))

print('Random matrix :\n', A)
# A = (A + A.T) / 2
#
# Ainv = np.linalg.inv(A)
diag_sum = np.sum(np.abs(A), axis=1)
np.fill_diagonal(A, diag_sum)

print('invertible  matrix ')
print(A)

# Eigenvalues and eigenvectors of A
# scipy.linalg.eig(A)
eigenval, eigenvec = np.linalg.eig(A)

print('Eigenvalues :\n', eigenval)
print('Eigenvectors :\n', eigenvec)

# Generate inverse matrix from eigenvectors
Q = np.linalg.inv(eigenvec)


# Generate diagonal matrix from eigenvalues
L= np.diag(eigenval)

# Reconstruct the matrix
# A2 = Q.dot(L).dot(Q.T)
# print('matrix original :\n', A2)
A3 = eigenvec.dot(L.dot(Q))
print('matrix original from eigen val and vac :\n', A3)

# eign_vector_inverse = np.linalg.inv(eigenvec)
print(np.allclose(A,A3))

# Verify that A2 is equal to A
# np.allclose(A2, A)
# print('they are true')
# True