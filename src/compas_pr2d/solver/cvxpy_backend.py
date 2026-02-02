import cvxpy as cp
from dataclasses import dataclass
import numpy as np
from problem.helpers import print_matrix


@dataclass
class Results:
    U: np.ndarray
    status: str
    objective: float


def solve_cvxpy(
    *, A_ub, b_ub, A_eq, b_eq, c, solver=cp.MOSEK, verbose=False
) -> Results:
    nvar = A_ub.shape[1]
    U = cp.Variable(nvar)

    # print("U Matrix:")
    # print_matrix(U)
    # print("A_ub Matrix:")
    # print_matrix(A_ub)
    # print("b_ub Matrix:")
    # print_matrix(b_ub)
    # print("A_eq Matrix:")
    # print_matrix(A_eq)
    # print("b_eq Matrix:")
    # print_matrix(b_eq)
    # print("c Matrix:")
    # print_matrix(c)

    constraints = [A_ub @ U >= b_ub, A_eq @ U == b_eq]
    objective = cp.Minimize(-(c @ U))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)
    return Results(U=U.value, status=prob.status, objective=prob.value)
