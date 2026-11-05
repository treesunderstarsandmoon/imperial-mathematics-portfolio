import numpy as np
import cla_utils
import matplotlib.pyplot as plt
from cla_utils.exercises8 import hessenberg
from cla_utils.exercises9 import inverse_it
from cla_utils.exercises10 import GMRES
# from cla_utils.exercises3  import solve_U
# from cla_utils.exercises3 import householder_ls


# question 1
def triQR(A, maxit, tol, use_zero_shift=False):
    # compute mu
    history = []
    m, n = A.shape
    for _ in range(maxit):
        if use_zero_shift:
            mu = 0
        else:
            delta = (A[m-2, m-2]-A[m-1, m-1])/2
            if (A[m-2, m-2]-A[m-1, m-1])/2 != 0:
                sig = np.sign(delta)
            else:
                sig = 1
            denominator = (np.abs(delta) + np.sqrt(delta**2+A[-1, -2]**2))
            mu = A[m-1, m-1]-(sig*A[m-1, m-2]**2)/denominator
        Q, R = np.linalg.qr(A - mu * np.eye(m))  # QR decomposition of A - mu*I
        A = R @ Q + mu * np.eye(m)
        history.append(np.abs(A[-1, -2]))
        if np.abs(A[-1, -2]) < tol:
            print()
            break
    return A, history
# print(triQR( np.array([
    [4, 1, 2],
    [1, 3, 0],
    [2, 0, 5]
# ]),10,0.001,use_zero_shift=False)[1])
# print(triQR( np.array([
    [4, 1, 2],
    [1, 3, 0],
    [2, 0, 5]
# ]),10,0.001,use_zero_shift=True)[1])


def random_symmetric_matrix(size, seed=None):
    if seed:
        np.random.seed(seed)
    A = np.random.randn(size, size)
    return (A + A.T) / 2

# Comparison and Plotting


def compare_shifts(matrix_size, maxit=50, tol=1e-6):
    A = random_symmetric_matrix(matrix_size, seed=42)
    _, history_wilkinson = triQR(A.copy(), maxit, tol, use_zero_shift=False)
    _, history_zero = triQR(A.copy(), maxit, tol, use_zero_shift=True)

    plt.figure(figsize=(10, 6))
    plt.semilogy(history_wilkinson, label="Wilkinson Shift", marker="o")
    plt.semilogy(history_zero, label="Zero Shift", marker="s")
    plt.title(f"Comparison of Convergence for Matrix Size {matrix_size}")
    plt.xlabel("Iteration")
    plt.ylabel(r"$|A_{m,m-1}|$ (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


compare_shifts(matrix_size=5)
compare_shifts(matrix_size=10)

# c.Reduce the matrix A to tridiagonal form


def find_eigenvalue(A):

    A = hessenberg(A)
    m = A.shape[0]
    eigenvalue_list = []
    for _ in range(m-1):
        # call triqr without shift
        A = triQR(A, 1000, 1e-8, use_zero_shift=True)[0]
        eigenvalue_list.append(A[-1, -1])
        A = A[:-1, :-1]
    eigenvalue_list.append(A[0, 0])
    return eigenvalue_list


A1 = np.random.randn(10, 10)
A = (A1.T + A1) / 2
normlist_1 = []
for i in range(10):
    x1 = inverse_it(A,
                    np.array(A[:, -1]),
                    find_eigenvalue(A)[i],
                    1e-8,
                    1000,
                    store_iterations=False)[1]
    x2 = find_eigenvalue(A)[i]
    normlist_1.append(np.linalg.norm(x1-x2))
print(normlist_1)
print("normlist_1 finished here")


A1 = np.random.randn(10, 10)
A = (A1.T + A1) / 2


def find_eigenvalue_without_mu(A):  # with iteration time
    iterate_time = 0
    A = hessenberg(A)
    m = A.shape[0]
    eigenvalue_list = []
    for _ in range(m-1):
        A, iter_history = triQR(A, 1000, 1e-8, use_zero_shift=True)
        eigenvalue_list.append(A[-1, -1])
        A = A[:-1, :-1]
        iterate_time += len(iter_history)
    eigenvalue_list.append(A[0, 0])
    iterate_time += 1
    return eigenvalue_list, iterate_time


def find_eigenvalue_use_mu(A):
    iterate_time = 0
    A = hessenberg(A)
    m = A.shape[0]
    eigenvalue_list = []
    for _ in range(m-1):
        A, iter_history = triQR(A, 1000, 1e-8, use_zero_shift=False)
        eigenvalue_list.append(A[-1, -1])
        A = A[:-1, :-1]
        iterate_time += len(iter_history)
    eigenvalue_list.append(A[0, 0])
    iterate_time += 1
    return eigenvalue_list, iterate_time


A1 = np.random.randn(10, 10)
A = (A1.T + A1) / 2
print(find_eigenvalue_without_mu(A)[1], find_eigenvalue_use_mu(A)[1])
B1 = np.random.randn(100, 100)
B = (B1.T + B1) / 2
print(find_eigenvalue_without_mu(B)[1], find_eigenvalue_use_mu(B)[1])

sizes = [10, 20, 50, 100, 200]

# Store eigenvalues
eigenvalues_without_mu_iterate = []
eigenvalues_with_mu_iterate = []

# Iterate through matrix sizes
for size in sizes:
    # Create a random symmetric matrix
    A1 = np.random.randn(size, size)
    A = (A1.T + A1) / 2  # Symmetrize

    # Compute iterate times
    time_without_mu = find_eigenvalue_without_mu(A)[1]
    time_with_mu = find_eigenvalue_use_mu(A)[1]
    # Store results
    eigenvalues_without_mu_iterate.append(time_without_mu)
    eigenvalues_with_mu_iterate.append(time_with_mu)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(sizes, eigenvalues_without_mu_iterate,
         marker='o', label='Without Shift (mu)')
plt.plot(sizes, eigenvalues_with_mu_iterate,
         marker='x', label='With Shift (mu)')
plt.xlabel('Matrix Size')
plt.ylabel('Eigenvalue')
plt.title('iteration vs. Matrix Size')
plt.legend()
plt.grid()
plt.show()

# question 2
n = 100
A_original = np.random.randn(n, n)  # Keep the original matrix
epsilons = [0.1, 0.001, 1e-6, 1e-8, 1e-10, 1e-12]
b = np.random.rand(n)

for epsilon in epsilons:
    A = A_original.copy()  # Use a copy of the original matrix
    A += 10*np.eye(n)
    A[0, :] = A[0, :] * epsilon
    x, nits, rnorms = GMRES(A, b, maxit=100, tol=1e-13,
                            return_residual_norms=True, return_residuals=False)
    plt.semilogy(rnorms, label=f'epsilon = {epsilon}')
plt.title('GMRES Residuals vs Iterations')
plt.legend()
plt.show()


def GMRES_modified(A, b, maxit, tol, return_residual_norms=False,
                   return_residuals=False, Jacobi=False, Gauss_Seidel=False):
    m = A.shape[0]
    x = np.zeros(m, dtype=A.dtype)
    Q = np.zeros((m, maxit + 1), dtype=A.dtype)
    H = np.zeros((maxit + 1, maxit), dtype=A.dtype)
    residuals = np.zeros((m, maxit))
    rnorms = np.zeros(maxit)
    iterations = maxit
    if Jacobi:
        A_hat = np.diag(np.diag(A))
        A_hat_inverse = np.diag(1/np.diag(A))
        b_hat = A_hat_inverse @ b
        q1 = b_hat/np.linalg.norm(b_hat)
        # q1=b/np.linalg.norm(b)
    elif Gauss_Seidel:
        A_hat = np.triu(A)
        b_hat = cla_utils.solve_U(A_hat, b)
        q1 = b_hat/np.linalg.norm(b_hat)
        # q1 = b/np.linalg.norm(b)
    else:
        b_hat = b.copy()
        q1 = b / np.linalg.norm(b)

    Q[:, 0] = q1
    for n in range(maxit):

        if Jacobi:
            # define v with A^v = A * qn
            v = A_hat_inverse @ A @ Q[:, n]
            # v = A @ Q[:, n]

        elif Gauss_Seidel:
            v = cla_utils.solve_U(
                A_hat, (A @ Q[:, n]).reshape(-1, 1)).flatten()
        else:
            v = A @ Q[:, n]

        for j in range(n + 1):
            H[j, n] = np.dot(Q[:, j].conj(), v)
            v -= H[j, n] * Q[:, j]
        H[n + 1, n] = np.linalg.norm(v)
        # Q[:, n + 1] = v / H[n + 1, n]
        if H[n + 1, n] > 0 and n + 1 < maxit:
            Q[:, n + 1] = v / H[n + 1, n]

        y = cla_utils.householder_ls(
            H[:n + 2, :n+1], np.linalg.norm(b_hat) * np.eye(n + 2)[:, 0])
        # print(y)
        # print(Q[:, :n+1])
        x = Q[:, :n + 1] @ y
        r = H[:n + 2, :n + 1] @ y - np.linalg.norm(b_hat) * np.eye(n + 2)[:, 0]
        rnorms[n] = np.linalg.norm(r)
        # residuals[:, n] = r
        # residuals[n] = r
        # print(r)
        if rnorms[n] < tol:
            iterations = n + 1
            break

    if return_residual_norms:
        if return_residuals:
            return (x, iterations, rnorms[:iterations], residuals[:iterations])
        else:
            return (x, iterations, rnorms[:iterations])
    else:
        if return_residuals:
            return (x, iterations, residuals[:iterations])
        else:
            return (x, n+1)


n = 100
A = np.random.randn(n, n)
epsilons = [0.1, 0.001, 1e-6, 1e-8, 1e-10, 1e-12]
b = np.random.rand(n)

for epsilon in epsilons:
    A1 = 1.0*A
    A1 += 10*np.eye(n)
    A1[0, :] = A1[0, :] * epsilon
    x, nits, rnorms = GMRES_modified(A1, b, maxit=200, tol=1e-13,
                                     return_residual_norms=True,
                                     return_residuals=False,
                                     Jacobi=False,
                                     Gauss_Seidel=True)
    # print(f'x is {x}')
    # print(f'nits is {nits}')
    # print(f'rnorms is {rnorms}')
    plt.semilogy(rnorms, label=f'epsilon = {epsilon}')

plt.xlabel('Iteration')
plt.ylabel('Residual Norm (log scale)')
plt.legend()
plt.title('Gauss_Seidel Residuals vs Iterations')
plt.grid(True)
plt.show()
for epsilon in epsilons:
    A1 = 1.0*A
    A1 += 10*np.eye(n)
    A1[0, :] = A1[0, :] * epsilon
    x, nits, rnorms = GMRES_modified(A1, b, maxit=200, tol=1e-13,
                                     return_residual_norms=True,
                                     return_residuals=False,
                                     Jacobi=True,
                                     Gauss_Seidel=False)
    # print(f'x is {x}')
    # print(f'nits is {nits}')
    # print(f'rnorms is {rnorms}')
    plt.semilogy(rnorms, label=f'epsilon = {epsilon}')

plt.xlabel('Iteration')
plt.ylabel('Residual Norm (log scale)')
plt.legend()
plt.title('Jacobi Residuals vs Iterations')
plt.grid(True)
plt.show()
