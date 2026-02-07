# src/prd/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import compas.geometry as cg
import numpy as np
import numpy.typing as npt


@dataclass
class Results:
    status: str
    objective: float
    U: npt.NDArray[np.float64]

    @property
    def U_blocks(self) -> npt.NDArray[np.float64]:
        if self.U.ndim != 1 or self.U.size % 3 != 0:
            raise ValueError("U array has incorrect shape.")
        return self.U.reshape((-1, 3))


class PR2DModel:
    """
    Piecewise Rigid Displacement Model for 2D block assemblies.

    Methods:
        from_polygons(polygons): Initialize model from a list of polygons.
        display_polygons(indices = False): Visualize the block geometry.
        set_boundary_edges(edges): Define block adjacency edges.
        set_displacement_bc(disp_data): Set displacement boundary conditions.
        set_force(loads): Apply external forces to blocks.
        solve(solver_opts): Solve the PRD problem.
        display_results(): Visualize the results of the analysis.
    """

    def __init__(self, *, gravity: float = -9.81) -> None:
        # inputs
        self.polygons: Optional[List[cg.Polygon]] = None
        self.mesh: Optional[cg.Mesh] = None
        self.n: Optional[List[Any]] = None  # (nv,2) coords (x,z)
        self.e: Optional[List[Any]] = None  # connectivity (optional)
        self.boundary_edges: Optional[List[Tuple[int, int]]] = None  # block adjacency
        self.boundaries: Optional[List[Tuple[int, int]]] = None  # block adjacency
        self.nnodes: Optional[int] = None

        self.disp_bcs: List[Tuple[int, float, float]] = []
        self.loads: Dict[int, Tuple[float, float, float]] = {}
        self.gravity = float(gravity)

        # derived / assembled
        self.contacts: Optional[list[Any]] = None
        self.A_ub: Optional[npt.NDArray[np.float64]] = None
        self.A_eq: Optional[npt.NDArray[np.float64]] = None
        self.c: Optional[npt.NDArray[np.float64]] = None
        self.b_ub: Optional[npt.NDArray[np.float64]] = None
        self.b_eq: Optional[npt.NDArray[np.float64]] = None

        # results
        self.results: Optional[Results] = None

    # -------------------------
    # Step 1: geometry init
    # -------------------------
    def from_mesh(self, mesh: cg.Mesh) -> "PR2DModel":
        if mesh is None:
            raise ValueError("Input mesh is None.")
        self.mesh = mesh

        from compas_pr2d.assembly.indexing import common_edges
        from compas_pr2d.assembly.indexing import mesh_to_vertices_edges

        self.n, self.e = mesh_to_vertices_edges(self.mesh)

        # self.contacts, self.boundaries = common_edges(self.mesh, self.boundary_edges)

        # self.intfn = len(self.contacts) + len(self.boundaries)

        # reset downstream (because geometry changed)
        self.reset_downstream()

        return self

    def from_polygons(self, polygons: List[cg.Polygon]) -> "PR2DModel":
        self.polygons = polygons

        # remap immediately (as you want)
        # from assembly.indexing import common_edges

        from compas_pr2d.assembly.indexing import mesh_polygon
        from compas_pr2d.assembly.indexing import mesh_to_vertices_edges

        self.mesh = mesh_polygon(self.polygons)
        self.n, self.e = mesh_to_vertices_edges(self.mesh)

        # contacts
        # self.contacts, self.boundaries = common_edges(self.mesh)

        # self.intfn = len(self.contacts) + len(self.boundaries)

        # reset downstream (because geometry changed)
        self.reset_downstream()
        return self

    def display_mesh(self, bindex=True, nindex=False) -> None:
        self._require_geom()  # Checks if geometry is initialized already
        from compas_pr2d.viz.prd_view import show_mesh

        show_mesh(self.mesh, bindex=bindex, nindex=nindex, n=self.n, e=self.e)

    def display_mesh_contact(
        self,
        bindex=True,
    ) -> None:
        from compas_pr2d.viz.prd_view import show_mesh_with_contact

        show_mesh_with_contact(self.mesh, bindex=bindex, n=self.n, e=self.all_contacts)

    # -------------------------
    # Step 2: connectivity + assembly
    # -------------------------
    def set_boundary_edges(self, boundary_edges: List[int] = None, display: bool = True) -> "PR2DModel":
        self._require_geom()  # Checks if geometry is initialized already
        # assemble contacts + matrices now
        self.assigned_boundary_edges = boundary_edges
        self._assemble_problem()
        if display:
            self.display_mesh_contact(bindex=True)
        return self

    def _assemble_problem(self) -> None:
        """Compute contacts + A-matrices and store them."""

        self._require_geom()
        from compas_pr2d.assembly.indexing import common_edges

        self.contacts, self.boundaries = common_edges(self.mesh, self.assigned_boundary_edges)
        self.intfn = len(self.contacts) + len(self.boundaries)
        self.all_contacts = self.contacts + self.boundaries
        # Matrix Assembly (A_ub, A_eq)
        from compas_pr2d.assembly.matrices import assemble_A_eq
        from compas_pr2d.assembly.matrices import assemble_A_ub
        from compas_pr2d.assembly.matrices import assemble_boundary_elements

        self.A_ub = assemble_A_ub(
            polygons=self.mesh,
            interface=self.all_contacts,
            nodes=self.n,
            nintf=self.intfn,
        )
        self.A_eq = assemble_A_eq(
            polygons=self.mesh,
            interface=self.all_contacts,
            nodes=self.n,
            nintf=self.intfn,
        )

        # assembling boundary elements A_ub_hat and A_eq_hat
        self.A_ub, self.A_eq = assemble_boundary_elements(
            self.mesh,
            self.n,
            self.intfn,
            self.all_contacts,
            self.A_ub,
            self.A_eq,
            len(self.contacts),
        )

        # RHS must be rebuilt later (BCs/loads)
        self.b_ub = np.zeros(2 * self.intfn)
        self.b_eq = np.zeros(2 * self.intfn)
        self.c = np.zeros(3 * self.mesh.number_of_faces())
        self.results = None

    # -------------------------
    # Step 3: BCs and loads
    # -------------------------

    def set_force(self, loads: Optional[Dict[int, Tuple[float, float, float]]] = None) -> "PR2DModel":
        self._require_geom()
        if loads is not None:
            self.loads = {int(k): tuple(map(float, v)) for k, v in loads.items()}
        # self._assemble_rhs()
        return self

    def assign_bc(self, disp_data: Sequence[Sequence[float]]) -> "PR2DModel":
        self._require_geom()
        self._require_matrices()
        self.disp_data = disp_data
        self._assemble_rhs()
        return self

    def _assemble_rhs(self) -> None:
        """Compute and store c, b_ub, b_eq."""
        self._require_geom()
        self._require_matrices()
        assert self.polygons is not None and self.A_ub is not None and self.A_eq is not None

        from compas_pr2d.assembly.rhs import build_force_vector
        from compas_pr2d.assembly.rhs import build_rhs_vectors
        from compas_pr2d.assembly.rhs import set_sw_force

        # self.c = build_force_vector(m=m, gravity=self.gravity, user_loads=self.loads)

        self.c = set_sw_force(c=self.c, magnitude=-abs(self.gravity))

        self.b_ub, self.b_eq = build_rhs_vectors(
            b_ub=self.b_ub,
            b_eq=self.b_eq,
            disp_data=self.disp_data,
            nodes=self.n,
            edge=self.all_contacts,
        )

        self.results = None

    # -------------------------
    # Step 4: solve + display
    # -------------------------
    def solve(self, dual: bool = False, mu: float = 0.2) -> Results:
        self._require_geom()
        self._require_matrices()

        # if RHS not ready yet, build it with defaults
        if self.c is None or self.b_ub is None or self.b_eq is None:
            self._assemble_rhs()

        from compas_pr2d.solver.cvxpy_backend import solve_cvxpy
        from compas_pr2d.solver.cvxpy_backend import solve_cvxpy_dual

        if dual:
            self.results = solve_cvxpy(
                self.mesh,
                A_ub=self.A_ub,
                b_ub=self.b_ub,
                A_eq=self.A_eq,
                b_eq=self.b_eq,
                c=self.c,
            )

            self.results = solve_cvxpy_dual(
                self.mesh,
                A_ub=self.A_ub,
                b_ub=self.b_ub,
                A_eq=self.A_eq,
                b_eq=self.b_eq,
                c=self.c,
                mu=mu,
                results=self.results,
                verbose=False,
            )

        else:
            self.results = solve_cvxpy(
                self.mesh,
                A_ub=self.A_ub,
                b_ub=self.b_ub,
                A_eq=self.A_eq,
                b_eq=self.b_eq,
                c=self.c,
            )

        if self.results.status == "unbounded":
            raise ValueError("The problem is unbounded.")
        if self.results.status != "optimal":
            raise ValueError("The solver didn't find an optimal solution.")

        return None

    def display_results(self, dual=False) -> None:
        self._require_geom()
        if self.results is None:
            raise ValueError("No results. Call solve() first.")
        from compas_pr2d.viz.prd_result_viewer import show_results
        from compas_pr2d.viz.prd_view_dual import show_dual_results

        if dual:
            print("Showing results for dual problem")
            show_dual_results(self, fn_opt=self.results.fn, ft_opt=self.results.ft, scale=0.01)
        else:
            print("Showing results for primal problem")
            show_results(self.mesh, self.results)

    # -------------------------
    # guards
    # -------------------------
    def _require_geom(self) -> None:
        if not self.polygons or self.mesh is None:
            raise ValueError("Call from_polygons() first.")

    def _require_matrices(self) -> None:
        if self.A_ub is None or self.A_eq is None:
            raise ValueError("Call the problem assembly function first.")

    def reset_downstream(self) -> None:
        """Reset all downstream data (contacts, matrices, results)."""
        self.A_ub = None
        self.A_eq = None
        self.c = None
        self.b_ub = None
        self.b_eq = None
        self.results = None
