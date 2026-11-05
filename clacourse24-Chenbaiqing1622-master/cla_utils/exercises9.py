import numpy as np
import numpy.random as random

def get_A100():
    """
    Return A100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    return A


def get_B100():
    """
    Return B100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A[np.tril_indices(m, -2)] = 0
    return A


def get_C100():
    """
    Return C100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    return A


def get_D100():
    """
    Return D100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    A[np.tril_indices(m, -2)] = 0
    A[np.triu_indices(m, 2)] = 0
    return A


def get_A3():
    """
    Return A3 matrix for investigating power iteration.
    
    :return A3: a 3x3 numpy array.
    """

    return np.array([[ 0.76505141, -0.03865876,  0.42107996],
                     [-0.03865876,  0.20264378, -0.02824925],
                     [ 0.42107996, -0.02824925,  0.23330481]])


def get_B3():
    """
    Return B3 matrix for investigating power iteration.

    :return B3: a 3x3 numpy array.
    """
    return np.array([[ 0.76861909,  0.01464606,  0.42118629],
                     [ 0.01464606,  0.99907192, -0.02666057],
                     [ 0.42118629, -0.02666057,  0.23330798]])


def pow_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either 

    ||r|| < tol where

    r = Ax - lambda*x,

    or the number of iterations exceeds maxit.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of power iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing all \
    the iterates.
    :return lambda0: the final eigenvalue.
    """
    v=x0
    for k in range(maxit):
        w = np.dot(A, v)
        
        # Step 2: v_k = w / ||w||
        v_new = w / np.linalg.norm(w)
        # Step 3: lambda(k) = v_k^T * A * v_k
        lambda_k = np.dot(v_new.T, np.dot(A, v_new))
        if np.linalg.norm(v_new - v) < tol:
            print(f"Converged after {k} iterations.")
            return v_new,lambda_k
        v=v_new
    
    return v_new, lambda_k


def inverse_it(A, x0, mu, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, a maxit dimensional numpy array containing \
    all the iterates.
    """
    v=x0
    m,n=A.shape
    for i in range(maxit):
        w = np.linalg.solve(A - mu * np.eye(m), v)
        v_new=w / np.linalg.norm(w)
        lambda_k = np.dot(v_new.T, np.dot(A, v_new))
        if np.linalg.norm(v_new - v) < tol:
            return v_new,lambda_k
        v=v_new
    return v_new, lambda_k

def rq_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """
    v=x0 #should be a unit vector
    lambda_old = np.dot(v.T, np.dot(A, v))
    if store_iterations:
        all_vectors = [v]
        all_eigenvalues = [lambda_old]
    for k in range(1, maxit + 1):
        I = np.eye(A.shape[0])
        w = np.linalg.solve(A - lambda_old * I, v)
        v = w / np.linalg.norm(w)
        lambda_new = np.dot(v.T, np.dot(A, v))
        if store_iterations:
            all_vectors.append(v)
            all_eigenvalues.append(lambda_new)
        if np.abs(lambda_new - lambda_old) < tol:
            break
        lambda_old = lambda_new
    if store_iterations:
        return np.array(all_vectors), np.array(all_eigenvalues)
    else:
        return v, lambda_new
    

def pure_QR(A, maxit, tol):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance

    :return Ak: the result
    """
    Ak = A
    for k in range(maxit):
        Q, R = np.linalg.qr(Ak)
        Ak = np.dot(R, Q)
        off_diagonal_norm = np.linalg.norm(Ak - np.tril(Ak), ord='fro') 
        
        if off_diagonal_norm < tol:
            print(f"Converged after {k+1} iterations")
            break
    return Ak


