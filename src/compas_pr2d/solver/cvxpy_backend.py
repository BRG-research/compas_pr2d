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
