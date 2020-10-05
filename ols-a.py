"""
Do OLS on the function FrankeFunction -
we have 2 variables so we need
1, x, x², x³, x⁴, x⁵, y, y², y³, y⁴, y⁵, xy, x²y, x³y, x⁴y, xy², x²y², x³y², xy³, x²y³, xy⁴

1+2+3+4+5+6 or in general, (n+1)*(n+2)/2 for nth degree polynomials.

Need to scale the data

"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


np.random.seed(1111)
# Make data, n meshpoints.
N = 100
#x = np.arange(0, 1, 1/N)
#y = np.arange(0, 1, 1/N)

x = np.random.rand(N)
y = np.random.rand(N)

x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def LinReg(X,Z):

    #X = Mesh2Des(x,y,p)
    #Z = Mesh2Vec(z)

    #print(X.shape)

    beta = np.linalg.inv(X.T @ X) @ (X.T) @ Z

    #Ztilde = X @ beta

    #print(beta)
    #print(Ztilde.shape)

    # Converting the prediction ztilde back to a meshgrid

    #n, m = x.shape[0],x.shape[1]

    #ztilde = Vec2Mesh(Ztilde,n, m)

    #print(ztilde)

    return beta

def Vec2Mesh(Z, n, m):
    """
    Takes a vector Z and converts it into an n x m matrix, row by row.
    """
    z = np.zeros((n,m))
    if n*m != len(Z):
        print("oh fuck")

    for i in range(n):
        z[i,:] = Z[m * i : m * i + m]
    return z

def Mesh2Vec(z):
    """
    Converts a meshgrid into a vector.
    """
    n, m = z.shape[0], z.shape[1]
    # Converting the function values to a vector
    Z = np.zeros(n * m)

    for i in range(n):
        for j in range(m):
            Z[m*i+j] = z[i,j]

    return Z

def Mesh2Des(x,y,p):
    """
    Converts a meshgrid x, y into a design matrix X of polynomials
    in x, y up to degree p.
    """
    n, m = x.shape[0], x.shape[1]

    # Number of predictors
    l = int((p+1) * (p+2) / 2)

    # First making a 3-dimensional design matrix utilizing the meshgrid. Coordinates are [y,x,degree].
    # The design matrix is an n x m x l matrix, where l is the number of predictors = (p+1)*(p+2)/2
    # for polynomials of degree p, with two variables.

    Y = np.zeros((n,m,l))
    k = 0

    for i in range(p+1):
        for j in range(p+1-i):
            Y[:,:,k] = (y**i)*(x**j)
            k += 1

    # Converting the 3-dimensional matrix to a 2-dimensional one, starting with y_0 for the first m rows spanning
    # x_0-x_m-1, then y_1 for another m rows, etc.

    # Design Matrix
    X = np.zeros((n * m, l))

    for i in range(n):
        for j in range(m):
            X[n*i+j,:] = Y[i,j,:]

    return X

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def R2(y_data, y_model):
    n = np.size(y_model)
    return 1 - np.sum((y_data-y_model)**2)/np.sum((y_data-np.sum(y_data/n))**2)

def makePred(X, p,beta):
    """
    Makes a prediction with a given predictor vector beta on the mesh
    grid x, y.
    """
    #(n, m) = x.shape
    #X = Mesh2Des(x,y,p)
    #print(n)
    #print(m)
    Z_pred = X @ beta
    #z_pred = Vec2Mesh(Z_pred, n, m)
    #print(f"Predshape = {z_pred.shape}")
    return Z_pred

p = 5
sigma = 0.1
noise = np.random.normal(0, sigma, (N,N))
z = FrankeFunction(x, y) + noise
#print(noise)


X = Mesh2Des(x,y,p)
Z = Mesh2Vec(z)
X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.2)
#x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.01)

scaler = StandardScaler()
scaler.fit(X_train[:,1:])
X_train_scaled = np.ones(X_train.shape)
X_test_scaled = np.ones(X_test.shape)
X_train_scaled[:,1:] = scaler.transform(X_train[:,1:])
X_test_scaled[:,1:] = scaler.transform(X_test[:,1:])
#print(X_test_scaled)
"""
x_test = x
x_train = x
y_test = y
y_train = y
z_test = z
z_train = z
"""

beta = LinReg(X_train_scaled, Z_train)
Z_pred = makePred(X_test_scaled, p, beta)

np.set_printoptions(precision=3, suppress = True)
print(f"MSE = {MSE(Z_test,Z_pred):.3f}")
print(f"R² = {R2(Z_test,Z_pred):.3f}")
print(beta)
print(Z_test)
print(Z_pred)
#print(Mesh2Des(x_train,y_train,p)[1])
