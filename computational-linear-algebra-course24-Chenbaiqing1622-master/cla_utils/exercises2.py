import numpy as np


def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """
    n = v.shape[0]
    u=np.zeros(n)
    for i in range(n):
        u[i] = np.outer(Q[:,i],Q[:,i].conj())

    r=v-u.dot(v)
    return r, u


def solve_Q(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """
    m=b.shape[0]
    Q1=np.zeros((m, m))
    for i in range(m):
        Q1[i]=Q[:,i].conj
    x=Q1.dot(b)

    

    return x


def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """

    n=Q.shape[0]
    for i in range(n):
        P+=np.outer(Q[:,i],Q[:,i].conj())
    

    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an mxl-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U, for appropriate l.
    """
    V1=np.zeros((m, m))
    for i in range(m): 
        V1[i]=V[:,i].conj
    Q=np.eye(m)-V.dot(V1)
   

    return Q


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm, transforming A to Q in place and returning R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """
    m, n = A.shape
    
    R = np.zeros((n, n))
    for j in range(n):
        v_j = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(A[:, i].conj(), A[:, j])  
            v_j = v_j - R[i, j] * A[:, i]       

        R[j, j] = np.linalg.norm(v_j)
        A[:, j] = v_j / R[j, j]  

    return R

def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, transforming A to Q in place and returning
    R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """
    m,n=np.shape(A)

    R = np.zeros((n, n))

    for j in range(n):
        v_j = A[:, j]
        R[j, j] = np.linalg.norm(v_j)
        A[:, j] = v_j / R[j, j] 
        for k in range(j+1,n):
            R[j, k] = np.dot(A[:, j].conj(), A[:, k])
            A[:, k] -= R[j, k] * A[:, j]
    
    return R


def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is normalised and orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """
    m,n=A.shape[0],A.shape[1]
    
    

    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    A = 1.0*A
    R = np.eye(n, dtype=A.dtype)
    for i in range(n):
        Rk = GS_modified_get_R(A, i)
        A[:,:] = np.dot(A, Rk)
        R[:,:] = np.dot(R, Rk)
    R = np.linalg.inv(R)
    return A, R
