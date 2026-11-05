import numpy as np
import cla_utils
import matplotlib.pyplot as plt

def GMRES_modified(A, b, maxit, tol, return_residual_norms=False,
          return_residuals=False, Jacobi=False, Gauss_Seidel=False):
    m = A.shape[0]
    x = np.zeros(m, dtype=A.dtype)
    Q = np.zeros((m, maxit + 1), dtype=A.dtype)  
    H = np.zeros((maxit + 1, maxit), dtype=A.dtype)  
    residuals = np.zeros((m, maxit))  
    rnorms = np.zeros(maxit)
    iterations = maxit
    if Jacobi==True:
        A_hat = np.diag(np.diag(A))
        A_hat_inverse=np.diag(1/np.diag(A))
        b_hat = A_hat_inverse @ b
        q1=b_hat/np.linalg.norm(b_hat)
    elif Gauss_Seidel == True:
        A_hat = np.triu(A)
        b_hat = cla_utils.solve_U(A_hat, b)
        q1 = b_hat/np.linalg.norm(b_hat)
    else:
        b_hat = b.copy()
        q1 = b / np.linalg.norm(b)

    Q[:, 0] = q1

    for n in range(maxit):
        
        if Jacobi:
            # define v with A^v = A * qn
            v=A_hat_inverse @ A @ Q[:, n]
            
        elif Gauss_Seidel:
            v=cla_utils.solve_U(A_hat,A @ Q[:, n])
        else:
            v = A @ Q[:, n]  
        
        for j in range(n + 1):
            H[j, n] = np.dot(Q[:, j].conj(), v)  
            v -= H[j, n] * Q[:, j]  
        H[n + 1, n] = np.linalg.norm(v)
        Q[:, n + 1] = v / H[n + 1, n]

        y = cla_utils.householder_ls(H[:n + 2, :n+1], np.linalg.norm(b_hat) * np.eye(n + 2)[:, 0])
        #print(y)
        #print(Q[:, :n+1])
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


n = 10
A = np.random.randn(n, n)
epsilons = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
b = np.random.rand(n)

for epsilon in epsilons:
    A1 = 1.0*A
    A1[0, :] =A1[0, :] * epsilon
    x, nits, rnorms = GMRES_modified(A1, b, maxit=100, tol=1e-6, return_residual_norms=True,return_residuals=False, Jacobi=False, Gauss_Seidel=True)
    #print(f'x is {x}')
    #print(f'nits is {nits}')
    #print(f'rnorms is {rnorms}')
    plt.semilogy(rnorms, label=f'epsilon = {epsilon}')

plt.xlabel('Iteration')
plt.ylabel('Residual Norm (log scale)')
plt.legend()
plt.title('Jacobi Residuals vs Iterations')
plt.grid(True)
plt.show()