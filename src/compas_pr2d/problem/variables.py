import time

# from numpy import asarray
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from networkx import nodes
import numpy.typing as npt
import numpy as np

import compas.geometry as cg

from interaction import contact_2d_edge
from interaction.contact_2d import contact_2d_1edge


@dataclass
class Problem:
    """
    The ``Problem`` class stores the matrices used in the optimisation. These are listed as parameters of the class and described below.

    Parameters
    ----------

    None

    Attributes
    ----------
    # Geometry and Connectivity

    n : array(m x 4)
        The node matrix
    e : array(m x 4)
        The edge connectivity matrix
    m : int
        The number of blocks
    ne : int
        The number of edges
    nv : int
        The number of vertices
    polygons : array(m x 4)
        The block polygon's connectivity matrix
    intf : list
        The list of interfaces between blocks (Interface i, between block j and k: intf[i] = (j,k))
    intfn: int
        The total number of interfaces
    intfcor: Array (intf x 4)
        The coordinates of the interface extremities (intfcor[i] = (x1,y1,x2,y2) for interface i between block j and k)
    intfedge: list
        The list of edges corresponding to each interface (intfedge[i] = (n1,n2) for interface i between nodes j and k)
    C : array(n x 3)
        The external force matrix
    U : array(m x 3)
        The Displacement matrix in X, Y and Rotation
    A_ub : array(m x (2*intf))
        The Unilateral restriction matrix
    A_eq : array(m x (2*intf))
        The no sliding equation matrix
    """

    n: Optional[npt.NDArray] = None
    e: Optional[npt.NDArray] = None
    m: Optional[int] = None
    ne: Optional[int] = None
    nv: Optional[int] = None
    polygons: Optional[List[cg.Polygon]] = None
    intf: Optional[List[Any]] = None
    intfn: Optional[int] = None
    intfcor: Optional[npt.NDArray] = None
    intfedge: Optional[List[Any]] = None

    C: Optional[npt.NDArray] = None
    U: Optional[npt.NDArray] = None
    A_ub: Optional[npt.NDArray] = None
    A_eq: Optional[npt.NDArray] = None

    # Attributes
    @classmethod
    def from_polygons(cls, polygons) -> "Problem":
        """Create a Problem from a set of polygons.

        This is the primary constructor for creating a Problem instance. It initializes
        all matrices and vectors needed for solving the stability problem.

        Parameters
        ----------
        form : :class:`~compas.geometry.diagrams.Polygons`
            The Polygons containing the geometry.

        Returns
        -------
        :class:`Problem`
            A fully initialized Problem instance with all necessary matrices.

        Examples
        --------
        >>> problem = Problem.from_polygons(form)
        >>> problem.m  # number of edges
        >>> problem.n  # number of vertices

        """
        # Mapping
        nv = 0
        ne = 0

        m = len(polygons)

        C = np.zeros((3 * m))
        U = np.zeros((3 * m))
        intf = contact_2d_edge(polygons)
        intf = contact_2d_1edge(polygons)
        intfn = len(intf)
        intfedge = None

        n, e = cls._remap_polygons(polygons, intf)

        cls._assemble_A_ub(
            polygons,
            m=m,
            intfcor=n,
            interface=intf,
        )

        cls._assemble_A_eq(
            polygons,
            m=m,
            intfcor=n,
            interface=intf,
        )

        BC_1 = list(polygons[0].edges())[0]
        BC_2 = list(polygons[-1].edges())[3]

        cls._set_BC(
            polygons[0],
            0,
            intfcor=n,
            Edge_id=BC_1,
            # Edge_id=(0, 1),
            A_ub=cls.A_ub,
            A_eq=cls.A_eq,
            interface_id=0,
        )
        cls._set_BC(
            polygons[-1],
            m - 1,
            intfcor=n,
            Edge_id=BC_2,
            # Edge_id=(2, 3),
            A_ub=cls.A_ub,
            A_eq=cls.A_eq,
            interface_id=intfn + 1,
        )
        # Note that the number of interfaces excludes the BCs, thus intfn + 1
        q = 2
        bi, bj, (iA1, iA2) = intf[q]

        Pi_a = polygons[bi].vertex[iA1]
        Pi_b = polygons[bi].vertex[iA2]

        Ai = n[bi * 4 + iA1]
        Bi = n[bj * 4 + iA2]

        print("poly.vertex a:", Pi_a["x"], Pi_a["z"], " intfcor:", Ai)
        print("poly.vertex b:", Pi_b["x"], Pi_b["z"], " intfcor:", Bi)

        for key, polyg in enumerate(polygons):
            nv += len(polyg.vertex)
            ne += len(list(polyg.edges()))

        # Create Problem instance
        return cls(
            n=n,
            e=e,
            m=m,
            ne=ne,
            nv=nv,
            polygons=polygons,
            intf=intf,
            intfn=intfn,
            intfedge=intfedge,
            C=C,
            U=U,
            A_ub=cls.A_ub,
            A_eq=cls.A_eq,
        )

    # =============================================================================
    # Remap Polygons
    # =============================================================================

    @classmethod
    def _remap_polygons(cls, polygons: List[cg.Polygon], interfaces: List[Any]):
        """
        Make sure the polygons are properly indexed
        """
        polygons_remap = np.copy(polygons)

        m = len(polygons)
        n = np.zeros((m * 4, 2), dtype=float)
        e = np.zeros((m, 4), dtype=int)
        for idx, poly in enumerate(polygons):
            polygons_remap[idx] = poly.copy()
            # vertices = poly.to_vertices_and_faces()
            # for i, vertex in enumerate(vertices):
            #     n[idx * 4 + i, 0] = vertex[0]
            #     n[idx * 4 + i, 1] = vertex[2]
            #     e[idx, i] = idx * 4 + i + 1  # +1 for 1-based indexing
            if len(poly.vertex) != 4:
                raise ValueError(
                    f"Polygon {idx} has {len(poly.vertex)} vertices; expected 4."
                )

            for k in range(4):
                vk = poly.vertex[k]
                n[idx * 4 + k, 0] = vk["x"]
                n[idx * 4 + k, 1] = vk["z"]
                e[idx, k] = idx * 4 + k  # keep 0-based unless you truly need 1-based

        return n, e

    # =============================================================================
    # Assemble Matrices
    # =============================================================================

    @classmethod
    def _assemble_A_ub(
        cls,
        polygons: List[cg.Polygon],
        m: int,
        interface: List[Any],
        intfcor,
    ):
        """
        Assembles the A ub Matrix
        """
        intfn = len(interface)
        A_ub = np.zeros((2 * (intfn + 2), 3 * m))

        for i, intf in enumerate(interface):
            bi, bj, (iA1, iA2) = intf

            Ax, Az = intfcor[bi * 4 + iA1]
            Bx, Bz = intfcor[bi * 4 + iA2]

            Edge_x = Bx - Ax
            Edge_y = Bz - Az

            edge = cg.Vector(Edge_x, 0, Edge_y)
            edge.unitize()

            nx = -edge[2]
            nz = edge[0]

            ci = polygons[bi].centroid()
            cj = polygons[bj].centroid()

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

            rA = 2 * i + 2
            rB = 2 * i + 3

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

        cls.A_ub = A_ub
        return None

    # =============================================================================

    @classmethod
    def _assemble_A_eq(
        cls,
        polygons: List[cg.Polygon],
        m: int,
        interface,
        intfcor,
    ):
        """
        Assembles the A eq Matrix
        """
        intfn = len(interface)
        A_eq = np.zeros((2 * (intfn + 2), 3 * m))

        for i, intf in enumerate(interface):
            bi, bj, (iA1, iA2) = intf

            Ax, Az = intfcor[bi * 4 + iA1]
            Bx, Bz = intfcor[bi * 4 + iA2]

            Edge_x = Bx - Ax
            Edge_z = Bz - Az
            edge = cg.Vector(Edge_x, 0, Edge_z)
            edge.unitize()

            tx = edge[0]
            tz = edge[2]

            ci = polygons[bi].centroid()
            cj = polygons[bj].centroid()

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

            rA = 2 * i + 2
            rB = 2 * i + 3

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

            polygons[bj].eq = 1

        cls.A_eq = A_eq
        return None

    @classmethod
    def _set_BC(
        cls,
        polygon: cg.Polygon,
        Block_id: int,
        intfcor,
        Edge_id,
        A_ub,
        A_eq,
        interface_id,
    ):
        """
        Apply Boundary Conditions to the Matrices
        """
        print(f"Setting BC for Block ID: {Block_id + 1}")
        bi = Block_id
        b2_idx = (bi) * 4
        (iA1, iA2) = Edge_id
        Ax, Az = intfcor[b2_idx + iA1]
        Bx, Bz = intfcor[b2_idx + iA2]

        Edge_x = Bx - Ax
        Edge_z = Bz - Az

        edge = cg.Vector(Edge_x, 0, Edge_z)
        edge.unitize()

        tx = edge[0]
        tz = edge[2]
        nx = -edge[2]
        nz = edge[0]

        ci = polygon.centroid()

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

        print(f"Normal Vector: ({nx}, {nz})")
        print(f"Tangential Vector: ({tx}, {tz})")

        def alpha(Px, Py, ck, tnx, tny):
            return -tnx * (Py - ck[2]) + tny * (Px - ck[0])

        # alpha(P;n) = -tx*(yP - y0) + ty*(xP - x0)
        Ax, Az = Ax, Az
        Bx, Bz = Bx, Bz

        ai_A_n = alpha(Ax, Az, ci, nx, nz)
        ai_B_n = alpha(Bx, Bz, ci, nx, nz)

        ai_A_t = alpha(Ax, Az, ci, tx, tz)
        ai_B_t = alpha(Bx, Bz, ci, tx, tz)

        rA = 2 * interface_id
        rB = 2 * interface_id + 1

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

        cls.A_eq = A_eq
        cls.A_ub = A_ub

        return None
