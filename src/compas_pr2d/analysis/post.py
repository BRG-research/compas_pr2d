# prd/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, List, Dict
import numpy as np
import numpy.typing as npt

import compas.geometry as cg


# ----------------------------
# Small typed containers
# ----------------------------

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class EdgeBC:
    """Boundary condition applied on a specific block edge.

    Parameters
    ----------
    block_id : int
        Block index in polygons list.
    edge : tuple[int, int]
        Local vertex indices (iA1, iA2) defining the edge.
    ub_value : (float, float)
        Values for the two rows in your unilateral BC RHS (b_ub).
    eq_value : (float, float)
        Values for the two rows in your equality BC RHS (b_eq).
    interface_id : int
        Which "interface slot" this BC occupies in the matrices (your current scheme).
    """
    block_id: int
    edge: Tuple[int, int]
    ub_value: Vec2 = (0.0, 0.0)
    eq_value: Vec2 = (0.0, 0.0)
    interface_id: int = 0


@dataclass
class Results:
    status: str
    objective: float
    U: npt.NDArray[np.float64]

    @property
    def U_blocks(self) -> npt.NDArray[np.float64]:
        # (m,3) view for convenience
        return self.U.reshape((-1, 3))


@dataclass
class AssembledProblem:
    """Everything the solver backend needs (no CVXPY objects)."""
    m: int
    intfn: int
    n: npt.NDArray[np.float64]      # node coords
    e: npt.NDArray[np.int64]        # connectivity (optional)
    interfaces: list[Any]           # output of contact detection
    A_ub: npt.NDArray[np.float64]
    A_eq: npt.NDArray[np.float64]
    c: npt.NDArray[np.float64]
    b_ub: npt.NDArray[np.float64]
    b_eq: npt.NDArray[np.float64]


# ----------------------------
# PRDModel facade
# ----------------------------

class PRDModel:
    """High-level user-facing model.

    Responsibilities:
    - store user inputs (geometry, loads, BCs, solver options)
    - orchestrate: detect contacts -> assemble matrices -> build RHS -> solve
    - provide results as attributes

    NOT responsible for:
    - contact detection math
    - matrix assembly math
    - CVXPY formulation details
    """

    def __init__(self) -> None:
        # Primary inputs
        self.polygons: Optional[List[cg.Polygon]] = None

        # User-defined inputs
        self._loads: Dict[int, Vec3] = {}     # block_id -> (fx, fz, mz) in your ordering
        self._edge_bcs: List[EdgeBC] = []

        # Options
        self.solver_name: str = "MOSEK"
        self.verbose: bool = False

        # Results
        self.results: Optional[Results] = None

        # Derived caches (invalidate on changes)
        self._interfaces: Optional[list[Any]] = None
        self._n: Optional[npt.NDArray[np.float64]] = None
        self._e: Optional[npt.NDArray[np.int64]] = None
        self._A_ub: Optional[npt.NDArray[np.float64]] = None
        self._A_eq: Optional[npt.NDArray[np.float64]] = None

    # ----------------------------
    # Construction / mutation API
    # ----------------------------

    def from_polygons(self, polygons: Sequence[cg.Polygon]) -> "PRDModel":
        self.polygons = list(polygons)
        self._invalidate_all()
        return self

    def clear_loads(self) -> None:
        self._loads.clear()
        self._invalidate_rhs_only()

    def set_force(self, block_id: int, force: Vec3) -> None:
        """Set block load/force contribution in global c vector (3 dof per block)."""
        self._require_geometry()
        self._loads[int(block_id)] = tuple(map(float, force))  # type: ignore
        self._invalidate_rhs_only()

    def set_forces(self, loads: Dict[int, Vec3]) -> None:
        self._require_geometry()
        self._loads = {int(k): tuple(map(float, v)) for k, v in loads.items()}  # type: ignore
        self._invalidate_rhs_only()

    def clear_bcs(self) -> None:
        self._edge_bcs.clear()
        self._invalidate_rhs_only()  # RHS changes, matrices may also change if BC modifies A (your scheme)
        self._invalidate_matrices_only()

    def add_edge_bc(self, bc: EdgeBC) -> None:
        self._require_geometry()
        self._edge_bcs.append(bc)
        # BCs in your current approach modify matrices *and* RHS.
        self._invalidate_rhs_only()
        self._invalidate_matrices_only()

    # ----------------------------
    # Solve pipeline
    # ----------------------------

    def assemble(self) -> AssembledProblem:
        """Assemble matrices and RHS (no solve)."""
        self._require_geometry()
        m = len(self.polygons)

        # 1) Interfaces / contacts
        if self._interfaces is None:
            self._interfaces = self._detect_contacts(self.polygons)

        interfaces = self._interfaces
        intfn = len(interfaces)

        # 2) Remap / node coords
        if self._n is None or self._e is None:
            self._n, self._e = self._remap_polygons(self.polygons)

        n = self._n
        e = self._e

        # 3) Assemble A matrices (without BCs first)
        if self._A_ub is None or self._A_eq is None:
            A_ub = self._assemble_A_ub(self.polygons, m=m, interface=interfaces, intfcor=n)
            A_eq = self._assemble_A_eq(self.polygons, m=m, interface=interfaces, intfcor=n)

            # Apply edge BCs into matrices (your current approach)
            for bc in self._edge_bcs:
                A_ub, A_eq = self._apply_edge_bc(
                    polygons=self.polygons,
                    block_id=bc.block_id,
                    intfcor=n,
                    edge_id=bc.edge,
                    A_ub=A_ub,
                    A_eq=A_eq,
                    interface_id=bc.interface_id,
                )

            self._A_ub, self._A_eq = A_ub, A_eq

        A_ub = self._A_ub
        A_eq = self._A_eq

        # 4) Build RHS vectors and force vector
        c = self._build_force_vector(m)
        b_ub, b_eq = self._build_rhs_vectors(A_ub.shape[0], A_eq.shape[0])

        # Apply BC values to RHS based on bc.interface_id (your scheme)
        for bc in self._edge_bcs:
            b_ub = self._set_bc_rows(b_ub, bc.interface_id, bc.ub_value)
            b_eq = self._set_bc_rows(b_eq, bc.interface_id, bc.eq_value)

        return AssembledProblem(
            m=m,
            intfn=intfn,
            n=n,
            e=e,
            interfaces=interfaces,
            A_ub=A_ub,
            A_eq=A_eq,
            c=c,
            b_ub=b_ub,
            b_eq=b_eq,
        )

    def solve(self) -> Results:
        """Assemble + solve; stores self.results and returns it."""
        assembled = self.assemble()
        res = self._solve_backend(
            A_ub=assembled.A_ub,
            b_ub=assembled.b_ub,
            A_eq=assembled.A_eq,
            b_eq=assembled.b_eq,
            c=assembled.c,
            solver_name=self.solver_name,
            verbose=self.verbose,
        )
        self.results = res
        return res

    # ----------------------------
    # Cache invalidation
    # ----------------------------

    def _invalidate_all(self) -> None:
        self.results = None
        self._interfaces = None
        self._n = None
        self._e = None
        self._A_ub = None
        self._A_eq = None

    def _invalidate_matrices_only(self) -> None:
        self.results = None
        self._A_ub = None
        self._A_eq = None

    def _invalidate_rhs_only(self) -> None:
        self.results = None
        # keep interfaces/n/e/A if only RHS changes

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _require_geometry(self) -> None:
        if self.polygons is None or len(self.polygons) == 0:
            raise ValueError("No polygons set. Call model.from_polygons(polygons) first.")

    def _build_force_vector(self, m: int) -> npt.NDArray[np.float64]:
        c = np.zeros((3 * m,), dtype=float)
        for bid, (fx, fz, mz) in self._loads.items():
            c[3 * bid + 0] = fx
            c[3 * bid + 1] = fz
            c[3 * bid + 2] = mz
        return c

    def _build_rhs_vectors(self, n_ub_rows: int, n_eq_rows: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        b_ub = np.zeros((n_ub_rows,), dtype=float)
        b_eq = np.zeros((n_eq_rows,), dtype=float)
        return b_ub, b_eq

    @staticmethod
    def _set_bc_rows(b: npt.NDArray[np.float64], interface_id: int, values: Vec2) -> npt.NDArray[np.float64]:
        rA = 2 * interface_id
        b[rA] = values[0]
        b[rA + 1] = values[1]
        return b

    # ----------------------------
    # Plug points (delegate to your modules)
    # Replace these with imports to your assembly/solver modules.
    # ----------------------------

    def _detect_contacts(self, polygons: list[cg.Polygon]) -> list[Any]:
        # from prd.interaction.contact_2d import contact_2d_1edge
        from interaction.contact_2d import contact_2d_1edge
        return contact_2d_1edge(polygons)

    def _remap_polygons(self, polygons: list[cg.Polygon]):
        # from prd.assembly.matrices import remap_polygons
        from problem.variables import Problem  # temporary, until you move it
        return Problem._remap_polygons(polygons, interfaces=[])

    def _assemble_A_ub(self, polygons, m: int, interface, intfcor):
        from problem.variables import Problem  # temporary, until you move it
        Problem._assemble_A_ub(polygons, m=m, interface=interface, intfcor=intfcor)
        return Problem.A_ub  # <-- you will remove this once you make it return arrays

    def _assemble_A_eq(self, polygons, m: int, interface, intfcor):
        from problem.variables import Problem  # temporary, until you move it
        Problem._assemble_A_eq(polygons, m=m, interface=interface, intfcor=intfcor)
        return Problem.A_eq

    def _apply_edge_bc(self, polygons, block_id: int, intfcor, edge_id, A_ub, A_eq, interface_id: int):
        from problem.variables import Problem  # temporary, until you move it
        Problem._set_BC(
            polygon=polygons[block_id],
            Block_id=block_id,
            intfcor=intfcor,
            Edge_id=edge_id,
            A_ub=A_ub,
            A_eq=A_eq,
            interface_id=interface_id,
        )
        return Problem.A_ub, Problem.A_eq

    def _solve_backend(self, *, A_ub, b_ub, A_eq, b_eq, c, solver_name: str, verbose: bool) -> Results:
        import cvxpy as cp

        nvar = A_ub.shape[1]
        U = cp.Variable(nvar)

        constraints = [
            A_ub @ U >= b_ub,
            A_eq @ U == b_eq,
        ]
        objective = cp.Maximize(c @ U)
        prob = cp.Problem(objective, constraints)

        # Map string to cp solver constant
        solver = getattr(cp, solver_name, None)
        if solver is None:
            raise ValueError(f"Unknown CVXPY solver '{solver_name}'.")

        prob.solve(solver=solver, verbose=verbose)
        return Results(status=prob.status, objective=float(prob.value), U=np.asarray(U.value, dtype=float))
