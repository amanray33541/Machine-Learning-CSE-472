import numpy as np

# n = np.random.randint(1,10)
n = int(input("Enter your value: "))
# create a random matrix
A = np.random.randint(-100,100,size=(n, n))
print('Random matrix :\n', A)

A = (A + A.T) // 2
print('symmetric :\n', A)

# symmetric A ko inverse niklayxi inverse ko transpose garda feri inverse nai aauxa
# while np.linalg.det(A) == 0:
#     A = np.random.randint(1,10,(n,n))
#     A = A + A.T
# print('matrix after symmetric :\n', A)
#
# Ainv = np.linalg.inv(A)
#
# print(Ainv)

# Eigenvalues and eigenvectors of A
# scipy.linalg.eig(A)
eigenval, eigenvec = np.linalg.eig(A)

print('Eigenvalues :\n', eigenval)
print('Eigenvectors :\n', eigenvec)

# Generate matrix from eigenvectors
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
