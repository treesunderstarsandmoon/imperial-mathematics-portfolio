import numpy as np
import numpy.random as random
from cla_utils.exercises3 import householder_ls

def arnoldi(A, b, k):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations

    :return Q: an mx(k+1) dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper \
    Hessenberg matrix
    """
    m = A.shape[0]
    Q = np.zeros((m, k + 1), dtype=complex)  
    H = np.zeros((k + 1, k), dtype=complex)  
    
    Q[:, 0] = b / np.linalg.norm(b)
    for n in range(k):
        v = A @ Q[:, n]
        for j in range(n + 1):
            H[j, n] = np.dot(Q[:, j].conj().T, v)
            v = v - H[j, n] * Q[:, j]
        
        H[n + 1, n] = np.linalg.norm(v)
        Q[:, n + 1] = v / np.linalg.norm(v)

    return Q, H
    

def GMRES(A, b, maxit, tol, return_residual_norms=False,
          return_residuals=False):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param return_residual_norms: logical
    :param return_residuals: logical

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """
    m = A.shape[0]
    x = np.zeros(m, dtype=A.dtype)
    
    q1 = b / np.linalg.norm(b)
    Q = np.zeros((m, maxit + 1), dtype=A.dtype)  
    H = np.zeros((maxit + 1, maxit), dtype=A.dtype)  

    Q[:, 0] = q1
    residuals = np.zeros((m, maxit))  
    rnorms = np.zeros(maxit)
    iterations = maxit

    for n in range(maxit):
        
        v = A @ Q[:, n]  
        
        for j in range(n + 1):
            H[j, n] = np.dot(Q[:, j].conj(), v)  
            v -= H[j, n] * Q[:, j]  
        H[n + 1, n] = np.linalg.norm(v)
        Q[:, n + 1] = v / H[n + 1, n]  
        y = householder_ls(H[:n + 2, :n+1], np.linalg.norm(b) * np.eye(n + 2)[:, 0])
        x = Q[:, :n + 1] @ y  
        r = A @ x - b
        rnorms[n] = np.linalg.norm(r)
        residuals[:, n] = r
        if rnorms[n] < tol:
            iterations = n + 1
            break
    
    if return_residual_norms == True:
        if return_residuals == True:
            return (x, iterations, rnorms[:iterations], residuals[:iterations])
        else:
            return (x, iterations, rnorms[:iterations])
    else:
        if return_residuals == True:
            return (x, iterations, residuals[:iterations])
        else:
            return (x, n+1)
   
def get_AA100():
    """
    Get the AA100 matrix.

    :return A: a 100x100 numpy array used in exercises 10.
    """
    AA100 = np.fromfile('AA100.dat', sep=' ')
    AA100 = AA100.reshape((100, 100))
    return AA100


def get_BB100():
    """
    Get the BB100 matrix.

    :return B: a 100x100 numpy array used in exercises 10.
    """
    BB100 = np.fromfile('BB100.dat', sep=' ')
    BB100 = BB100.reshape((100, 100))
    return BB100


def get_CC100():
    """
    Get the CC100 matrix.

    :return C: a 100x100 numpy array used in exercises 10.
    """
    CC100 = np.fromfile('CC100.dat', sep=' ')
    CC100 = CC100.reshape((100, 100))
    return CC100
