import numpy as np


def inf_norm(y):
    return np.max(np.abs(y))


def solve_by_simple_iteration(A, b, eps=1e-15):
    alpha_matrix = -(A.T / np.diag(A) - np.eye(len(A))).T
    beta = b / np.diag(A)
    result = beta.copy()
    iters = 0
    while inf_norm(A.dot(result) - b) > eps:
        result = alpha_matrix.dot(result) + beta
        iters += 1
    print(iters)
    return result


def solve_by_Gauss_Seidel(A, b, eps=1e-15):
    result = np.zeros(len(A))
    iters = 0
    while inf_norm(A.dot(result) - b) > eps:
        for i in range(len(A)):
            result[i] = (-A[i].dot(result) + b[i])/A[i, i] + result[i]
        iters += 1
    print(iters)
    return result


def solve_by_Jacobi(A, b, eps=1e-15):
    result = np.zeros(len(A))
    iters = 0
    D = np.diag(A)
    while inf_norm(A.dot(result) - b) > eps:
        result = ((-A.dot(result) + b + D*result) / D)
        iters += 1
    print(iters)
    return result
