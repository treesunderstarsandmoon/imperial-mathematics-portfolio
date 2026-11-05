import numpy as np
import cla_utils


def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """
    m, n = A.shape
    x = A[:, 0]
    if x[0] != 0:
        sig = np.sign(x[0])  
    else:
        sig=1
    v = x.copy()
    v[0] += sig * np.linalg.norm(x)  
    v /= np.linalg.norm(v)  
    Q1 = np.eye(m) - 2 * np.outer(v, v)
    A1 = Q1 @ A @ Q1.T.conj()
    
    return A1

    
    

def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """
    m, n = A.shape
    for k in range(m-2):
        x=A[k+1:m,k]
        sig = np.sign(x[0]) if x[0] != 0 else 1
        v = x.copy()
        v[0] += sig * np.linalg.norm(x)  # Modify the first entry
        v = v / np.linalg.norm(v)  # Normalize
        A[k+1:m, k:m] = A[k+1:m, k:m] - 2 * np.outer(v, np.dot(v.conj().T, A[k+1:m, k:m]))
        A[0:m, k+1:m] = A[0:m, k+1:m] - 2 * np.outer(np.dot(A[0:m, k+1:m], v), v.conj().T)
    return A



def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array
    
    :return Q: an mxm numpy array
    """
    m, n = A.shape
    Q = np.eye(m)
    for k in range(m-2):
        x=A[k+1:m,k]
        sig = np.sign(x[0]) if x[0] != 0 else 1
        v = x.copy()
        v[0] += sig * np.linalg.norm(x)  # Modify the first entry
        v /= np.linalg.norm(v)  # Normalize
        H = np.eye(m-k-1) - 2 * np.outer(v, v)
        A[k+1:m, k:m] -= 2 * np.outer(v, v.conj().T @ A[k+1:m, k:m])
        A[0:m, k+1:m] -= 2 * np.outer(A[0:m, k+1:m] @ v, v.conj().T)
        H_full = np.eye(m)
        H_full[k+1:m, k+1:m] = H
        Q=Q @ H_full
    return Q


def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvectors.

    :param H: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of H

    Do not change this function.
    """
    m, n = H.shape
    assert(m==n)
    assert(cla_utils.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V



def ev(A):
    """
    Given a matrix A, return the eigenvectors of A. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """
    A_copy = A.copy()
    H=hessenberg(A)#transform A to hessenberg
    Q=hessenbergQ(A_copy)
    y=hessenberg_ev(H)
    return Q @ y


