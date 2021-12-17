import numpy as np
from scipy import linalg
import sympy as sp
sp.init_printing(use_unicode=True)

mat_a = sp.Matrix([[-3, 2], [-1, 0]])
mat_a.eigenvects()

if __name__ == '__main__':
    mat = np.array([[2, -1, -1, 0, 0, 0, 0, 0, 0],
                    [-1, 3, -1, 0, 0, 0, 0, -1, 0],
                    [-1, -1, 3, -1, 0, 0, 0, 0, 0],
                    [0, 0, -1, 3, -1, -1, 0, 0, 0],
                    [0, 0, 0, -1, 3, -1, -1, 0, 0],
                    [0, 0, 0, -1, -1, 2, 0, 0, 0],
                    [0, 0, 0, 0, -1, 0, 3, -1, -1],
                    [0, -1, 0, 0, 0, 0, -1, 3, -1],
                    [0, 0, 0, 0, 0, 0, -1, -1, 2]])
    sp.init_printing(use_unicode=True)

    mat_a = sp.Matrix([[2, -1, -1, 0, 0, 0, 0, 0, 0],
                       [-1, 3, -1, 0, 0, 0, 0, -1, 0],
                       [-1, -1, 3, -1, 0, 0, 0, 0, 0],
                       [0, 0, -1, 3, -1, -1, 0, 0, 0],
                       [0, 0, 0, -1, 3, -1, -1, 0, 0],
                       [0, 0, 0, -1, -1, 2, 0, 0, 0],
                       [0, 0, 0, 0, -1, 0, 3, -1, -1],
                       [0, -1, 0, 0, 0, 0, -1, 3, -1],
                       [0, 0, 0, 0, 0, 0, -1, -1, 2]])

    # eig_val = np.linalg.eig(mat)
    # eig_val_2 = linalg.eig(mat)
    # eig_val[0].sort()
    # print(eig_val[0])
    # print(eig_val[1])
    print(mat_a.eigenvals())
    print(mat_a.eigenvects())
