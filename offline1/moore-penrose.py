import numpy as np

n = int(input("Enter your value of n: "))
m = int(input("Enter your value of m: "))
mat = np.random.randint(-100,100,size=(n,m))
print(mat)
U,d,V = np.linalg.svd(mat)

inv_mat1 = np.linalg.pinv(mat)
print(inv_mat1)

D  =  np.diag(d)
invD = np.linalg.inv(D)
plusD = np.concatenate((invD, np.array([[0,0]]).T),axis = 1)
tnrasposeV = V.T
inv_mat2 = tnrasposeV.dot(plusD.dot(U.T))

print(inv_mat2)
Aplus = np.dot(V.T, np.dot(plusD,U.T))
print(Aplus)

if np.allclose(inv_mat1, inv_mat2, atol = 10e-5):
    print('same matrix')
else:
    print('not same matrix')