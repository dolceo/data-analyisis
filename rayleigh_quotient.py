import numpy as np


def is_hermitian(matrix: np.array) -> bool:
    return np.array_equal(matrix, matrix.conjugate().T)


def rayleigh_quotient(A: np.array, v: np.array) -> float:
    v_star = v.conjugate().T
    return (v_star.dot(A).dot(v)) / (v_star.dot(v))


def tests() -> None:
    A = np.array([[2, 2 + 1j, 4], [2 - 1j, 3, 1j], [4, -1j, 1]])
    v = np.array([[1], [2], [3]])
    assert is_hermitian(A), f"{A} is not hermitian."
    print(rayleigh_quotient(A, v))

    A = np.array([[1, 2, 4], [2, 3, -1], [4, -1, 1]])
    assert is_hermitian(A), f"{A} is not hermitian."
    assert rayleigh_quotient(A, v) == float(3)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    tests()
