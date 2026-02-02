from typing import List, Any, Tuple
import numpy as np
import compas.geometry as cg
from compas.datastructures import Mesh

# =============================================================================
# Assemble Matrices
# =============================================================================


def assemble_A_ub(
    polygons: Mesh,
    interface: List[Any],
    nodes: List[cg.Point],
    nintf: int = None,
):
    """
    Assembles the A ub Matrix
    """
    m = polygons.number_of_faces()
    A_ub = np.zeros((2 * len(interface), 3 * m))

    for intf in interface:
        if len(intf) != 4:
            continue

        bi, bj, i, (iA1, iA2) = intf

        Ax, _, Az = nodes[iA1]
        Bx, _, Bz = nodes[iA2]
        Edge_x = Bx - Ax
        Edge_z = Bz - Az

        edge = cg.Vector(Edge_x, 0, Edge_z)
        edge.unitize()

        nx = -edge[2]
        nz = edge[0]

        ci = polygons.face_centroid(bi)
        cj = polygons.face_centroid(bj)

        dijx = cj[0] - ci[0]
        dijy = cj[2] - ci[2]

        if dijx * nx + dijy * nz < 0.0:
            nx *= -1.0
            nz *= -1.0

        def alpha_n(Px, Py, ck, nx, nz):
            return -nx * (Py - ck[2]) + nz * (Px - ck[0])

        # alpha(P;n) = -nx*(yP - y0) + nz*(xP - x0)

        ai_A = alpha_n(Ax, Az, ci, nx, nz)
        aj_A = alpha_n(Ax, Az, cj, nx, nz)
        ai_B = alpha_n(Bx, Bz, ci, nx, nz)
        aj_B = alpha_n(Bx, Bz, cj, nx, nz)

        rA = 2 * i
        rB = 2 * i + 1

        # Row at A: (u_j(A) - u_i(A))·n >= 0

        A_ub[rA, 3 * bi + 0] = -nx
        A_ub[rA, 3 * bi + 1] = -nz
        A_ub[rA, 3 * bi + 2] = -ai_A

        A_ub[rA, 3 * bj + 0] = +nx
        A_ub[rA, 3 * bj + 1] = +nz
        A_ub[rA, 3 * bj + 2] = +aj_A

        # Row at B: (u_j(B) - u_i(B))·n >= 0

        A_ub[rB, 3 * bi + 0] = -nx
        A_ub[rB, 3 * bi + 1] = -nz
        A_ub[rB, 3 * bi + 2] = -ai_B

        A_ub[rB, 3 * bj + 0] = +nx
        A_ub[rB, 3 * bj + 1] = +nz
        A_ub[rB, 3 * bj + 2] = +aj_B

    return A_ub

    # =============================================================================


def assemble_A_eq(
    polygons: Mesh,
    interface: List[Any],
    nodes: List[cg.Point],
    nintf: int = None,
):
    """
    Assembles the A eq Matrix
    """

    m = polygons.number_of_faces()

    m = polygons.number_of_faces()

    A_eq = np.zeros((2 * len(interface), 3 * m))

    for intf in interface:
        if len(intf) != 4:
            continue
        bi, bj, i, (iA1, iA2) = intf

        Ax, _, Az = nodes[iA1]
        Bx, _, Bz = nodes[iA2]

        Edge_x = Bx - Ax
        Edge_z = Bz - Az
        edge = cg.Vector(Edge_x, 0, Edge_z)
        edge.unitize()

        tx = edge[0]
        tz = edge[2]

        ci = polygons.face_centroid(bi)
        cj = polygons.face_centroid(bj)

        dijx = cj[0] - ci[0]
        dijy = cj[2] - ci[2]

        if dijx * tx + dijy * tz < 0.0:
            tx *= -1.0
            tz *= -1.0

        def alpha_t(Px, Py, ck, tx, ty):
            return -tx * (Py - ck[2]) + ty * (Px - ck[0])

        # alpha(P;n) = -tx*(yP - y0) + ty*(xP - x0)

        ai_A = alpha_t(Ax, Az, ci, tx, tz)
        ai_B = alpha_t(Bx, Bz, ci, tx, tz)
        aj_A = alpha_t(Ax, Az, cj, tx, tz)
        aj_B = alpha_t(Bx, Bz, cj, tx, tz)

        rA = 2 * i
        rB = 2 * i + 1

        # Row at A: (u_j(A) - u_i(A))·t = 0

        A_eq[rA, 3 * bi + 0] = -tx
        A_eq[rA, 3 * bi + 1] = -tz
        A_eq[rA, 3 * bi + 2] = -ai_A

        A_eq[rA, 3 * bj + 0] = +tx
        A_eq[rA, 3 * bj + 1] = +tz
        A_eq[rA, 3 * bj + 2] = +aj_A

        # Row at B: (u_j(B) - u_i(B))·t = 0
        A_eq[rB, 3 * bi + 0] = -tx
        A_eq[rB, 3 * bi + 1] = -tz
        A_eq[rB, 3 * bi + 2] = -ai_B

        A_eq[rB, 3 * bj + 0] = +tx
        A_eq[rB, 3 * bj + 1] = +tz
        A_eq[rB, 3 * bj + 2] = +aj_B

    return A_eq


def assemble_boundary_elements(
    polygon: Mesh,
    nodes: List[cg.Point],
    nintf: int = None,
    b_edge: List[Tuple[int, int, Tuple[int, int]]] = None,
    A_ub: np.ndarray = None,
    A_eq: np.ndarray = None,
    interf_n: int = 0,
):
    """
    Apply Boundary Conditions to the Matrices
    """
    for edge_id in b_edge:
        if len(edge_id) != 3:
            continue
        print(f"Assembling boundary elements for edge: {edge_id}")

        bi, k, (A, B) = edge_id

        Ax, _, Az = nodes[A]
        Bx, _, Bz = nodes[B]

        Edge_x = Bx - Ax
        Edge_z = Bz - Az

        edge = cg.Vector(Edge_x, 0, Edge_z)
        edge.unitize()

        tx = edge[0]
        tz = edge[2]
        nx = -edge[2]
        nz = edge[0]

        ci = polygon.face_centroid(bi)

        mid_x = 0.5 * (Ax + Bx)
        mid_z = 0.5 * (Az + Bz)
        vx = mid_x - ci[0]
        vz = mid_z - ci[2]
        if vx * nx + vz * nz > 0.0:
            nx *= -1.0
            nz *= -1.0

        if vx * tx + vz * tz < 0.0:
            tx *= -1.0
            tz *= -1.0

        def alpha(Px, Py, ck, tnx, tny):
            return -tnx * (Py - ck[2]) + tny * (Px - ck[0])

        # alpha(P;n) = -tx*(yP - y0) + ty*(xP - x0)

        ai_A_n = alpha(Ax, Az, ci, nx, nz)
        ai_B_n = alpha(Bx, Bz, ci, nx, nz)

        ai_A_t = alpha(Ax, Az, ci, tx, tz)
        ai_B_t = alpha(Bx, Bz, ci, tx, tz)

        rA = k * 2
        rB = k * 2 + 1
        print(f"Boundary edge rows: {rA}, {rB}")
        # A_ub BC - Block 1
        A_ub[rA, 3 * bi + 0] = +nx
        A_ub[rA, 3 * bi + 1] = +nz
        A_ub[rA, 3 * bi + 2] = +ai_A_n
        A_ub[rB, 3 * bi + 0] = +nx
        A_ub[rB, 3 * bi + 1] = +nz
        A_ub[rB, 3 * bi + 2] = +ai_B_n

        # A_eq BC - Block 1
        A_eq[rA, 3 * bi + 0] = +tx
        A_eq[rA, 3 * bi + 1] = +tz
        A_eq[rA, 3 * bi + 2] = +ai_A_t
        A_eq[rB, 3 * bi + 0] = +tx
        A_eq[rB, 3 * bi + 1] = +tz
        A_eq[rB, 3 * bi + 2] = +ai_B_t

    return A_ub, A_eq
