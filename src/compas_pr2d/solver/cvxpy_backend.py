from dataclasses import dataclass

import compas.geometry as cg
import cvxpy as cp
import numpy as np
from compas.datastructures import Mesh


@dataclass
class Results:
    U: np.ndarray
    U_n: np.ndarray
    status: str
    objective: float
    deformed_shape: list[Mesh] = None
    n_total: int = 0
    fn: np.ndarray = None
    ft: np.ndarray = None


def solve_cvxpy(mesh: Mesh, A_ub, b_ub, A_eq, b_eq, c, solver=cp.MOSEK, verbose=False) -> Results:
    U = cp.Variable(len(c))
    constraints = [A_ub @ U >= b_ub, A_eq @ U == b_eq]
    objective = cp.Minimize(-(c @ U))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)
    if prob.status != "optimal":
        raise ValueError("The solver didn't find an optimal solution.")

    deformed_shape, nodal_displacements, n_total = deform_polygon_linear(mesh, U.value, scale=1.0)

    return Results(
        U=U.value,
        U_n=nodal_displacements,
        deformed_shape=deformed_shape,
        status=prob.status,
        objective=prob.value,
        n_total=n_total,
        fn=None,
        ft=None,
    )


def deform_polygon_linear(poly: Mesh, U_solution, scale):
    n = poly.number_of_faces()
    U_new = U_solution.reshape((n, 3))
    deformed = []
    U_nodes = []
    n_total = 0
    for i in range(n):
        dx, dz, th = U_new[i]  # (Ux, Uz, theta)

        c = poly.face_centroid(i)
        cx, cz = c[0], c[2]
        new_pts = []
        for p in poly.face_vertices(i):  #
            x, z = poly.vertex[p]["x"], poly.vertex[p]["z"]
            x_new = x + dx - th * (z - cz)
            z_new = z + dz + th * (x - cx)
            new_pts.append(cg.Point(x_new, 0.0, z_new))
            U_nodes.append((x_new - x, 0.0, z_new - z))
            n_total += 1
        deformed.append((cg.Polygon(new_pts)).to_mesh())

    U_nodes = np.asarray(U_nodes)
    return deformed, U_nodes, n_total


def solve_cvxpy_dual(mesh: Mesh, A_ub, A_eq, c, b_ub, b_eq, mu, results, solver=cp.MOSEK, verbose=False) -> Results:

    m_ub, _ = A_ub.shape
    m_eq, _ = A_eq.shape
    mu = mu
    # Stacking the dual problem data as in the antonino's paper
    # As the CVXPY solver expects the objective function to be a single vector
    # not a decoupled fn and ft.

    # So all the data will be horitzontally stacked as follows:
    delta_n = -b_ub
    delta_t = -b_eq

    delta = np.hstack([delta_n, delta_t])  # (m_ub + m_eq,)
    A_T = np.hstack([A_ub.T, A_eq.T])  # (n, m_ub + m_eq)
    f = cp.Variable(m_ub + m_eq)

    fn = f[:m_ub]
    ft = f[m_ub:]

    constraints = [A_T @ f == c, fn <= 0]

    constraints += [
        ft <= -mu * (fn),
        ft >= mu * (fn),
    ]

    objective = cp.Minimize(-(delta @ f))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    print("status:", prob.status)
    print("objective:", prob.value)

    f_opt = f.value

    # Retrieveing the normal and tangential components of the force vector
    fn_opt = f_opt[:m_ub]  # normal component
    ft_opt = f_opt[m_ub:]  # tangential component

    if verbose:
        # Verify the optimality condition after decomposing the force vector.
        res = A_ub.T @ fn_opt + A_eq.T @ ft_opt - c
        print("||res||2 =", np.linalg.norm(res))

    return Results(
        U=results.U,
        U_n=results.U_n,
        status=results.status,
        objective=results.objective,
        deformed_shape=results.deformed_shape,
        n_total=results.n_total,
        fn=fn_opt,
        ft=ft_opt,
    )
