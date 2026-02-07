import numpy as np
import compas.geometry as cg


def build_rhs_vectors(b_ub, b_eq, disp_data, nodes, edge):
    """Sets the boundary conditions in the RHS vector b"""
    for disp_bc in disp_data:
        idx, fx_, fy_ = disp_bc
        if len(edge[idx]) != 3:
            print("Warning: Displacement BC specified is not on an interface edge.")
        _, _, (A, B) = edge[idx]
        print(f"Assigning displacement BC at interface {idx} on edge ({A}, {B})")

        if type(fx_) is tuple:
            fx = np.array([fx_[0], fx_[1]])
        else:
            fx = np.array([fx_, fx_])
        if type(fy_) is tuple:
            fy = np.array([fy_[0], fy_[1]])
        else:
            fy = np.array([fy_, fy_])

        # Converting to XY components can cause contradictions in the BC for now, unless i resolve another way
        # vx = nodes[A][0] - nodes[B][0]  # A_x - B_x
        # vz = nodes[A][1] - nodes[B][1]  # A_z - B_z
        # edge = cg.Vector(vx, 0, vz)
        # edge.unitize()
        # tx = edge[0]
        # tz = edge[2]
        # nx = -edge[2]
        # nz = edge[0]

        # fn = fx * nx + fy * nz
        # ft = fx * tx + fy * tz

        fn = fx
        ft = fy
        b_ub = assign_disp_bc(b_ub, idx, fn)
        b_eq = assign_disp_bc(b_eq, idx, ft)
    return b_ub, b_eq


def assign_disp_bc(b, idx, f):
    """Set displacement boundary conditions in the global RHS vectors b_ub and b_eq"""
    b[2 * idx] = -f[0]
    b[2 * idx + 1] = -f[1]
    return b


def build_force_vector(c, idx, vector):
    """Set force boundary conditions in the global force vector C"""
    c[idx * 3] = vector[0]
    c[idx * 3 + 1] = vector[1]
    c[idx * 3 + 2] = vector[2]
    return c


def set_sw_force(c, magnitude=0.0):
    """Set force boundary conditions in the global force vector C
    Arguements:
    c = [ux1, uy1, uth1, ux2, uy2, uth2, ..., uxn, uyn, uthn]
    Returns a new vector with all Uy entries set to `magnitude`.
    """
    v = np.asarray(c, dtype=float).reshape(-1)  # 1D
    if len(v) % 3 != 0:
        raise ValueError(f"Expected length 3*n, got {v.size}")
    out = v.copy()
    out.reshape(-1, 3)[:, 1] = magnitude  # column 1 == Uy

    return out
