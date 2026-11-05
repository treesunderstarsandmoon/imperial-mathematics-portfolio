import numpy as np


def householder(A):
    """
    Given a real mxn matrix A, reduce it to an upper triangular matrix R
    using Householder transformations, performing the reduction in-place.
    A is transformed to R.

    :param A: an mxn-dimensional numpy array (in-place transformation)
    :return: None (A is modified in-place to contain R)
    """
    m, n = A.shape

    for k in range(n):
        
        x = A[k:, k]
        if len(x)==0:
            break
        if x[0]==0:
            sig=1
        else:
            sig=np.sign(x[0])
        vk=sig*np.linalg.norm(x)*np.eye(1,m-k)[0]+x
        vk=vk/np.linalg.norm(vk)
        A[k:,k:]-=2*np.outer(vk,np.dot(np.conj(vk),A[k:,k:]))


def solve_U(U, b):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,...,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """
    m=U.shape[0]
    x = np.zeros_like(b)
    x[m-1]=b[m-1]/U[m-1,m-1]
    for i in range(m-2,-1,-1):
        x[i]= (b[i]-np.dot(U[i,i+1:],x[i+1:]))/U[i,i]
    return x
                     
    


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """

    m = A.shape[0]
    A_hat=np.hstack((A,b))
    householder(A_hat)
    R=A_hat[:m,:m]
    b_hat=A_hat[:m, m:]
    x=solve_U(R,b_hat)
    return x

    


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the full QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """
    m,n=A.shape
    I=np.identity(m)
    A_hat=np.hstack((A,I))
    householder(A_hat)
    R=A_hat[:,:n]
    Qstar=A_hat[:,n:]
    Q=np.conj(np.transpose(Qstar))
    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """
    n=A.shape[1]
    A_hat=np.hstack((A,b.reshape(-1,1)))
    householder(A_hat)
    R_hat=A_hat[:n,:n]
    b_hat=A_hat[:n,n]
    x=solve_U(R_hat,b_hat)
    return x
