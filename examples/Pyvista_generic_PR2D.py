# Note::::
# To run this example, you need to have pyvista installed.
# You can install it via pip:
# pip install pyvista

import numpy as np
import pyvista as pv
from compas.datastructures import Mesh
from pyvista.plotting.opts import ElementType
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout

from compas_pr2d import arch
from compas_pr2d import lintles
from compas_pr2d import wall
from compas_pr2d.prd import PR2DModel

height = 4
span = 10
n = 10
thickness = 0.5

model = PR2DModel()
arch_geometry = arch.construct_arch(height, span, n, thickness)
# arch_geometry = lintles.construct_lintles(height, span, n)
# arch_geometry = wall.construct_wall(length=5, height=4, nl=5, nh=4)

model.from_polygons(arch_geometry)
model.display_mesh(bindex=False, nindex=False)
model.set_boundary_edges(display=False)
disp_data = [
    [10, 0.0, -0.2],
]
# model.set_force()  # for now no loads, just self-weight which is default in the model (i will add as gravity to all blocks for now only)
model.assign_bc(disp_data)
model.solve()
model.display_results()
# -------------------------------------------------------------------------


# Visualize results using PyVista ------------------------------------------
Results = model.results
A = Results.deformed_shape
# print("----- PR2D Analysis Results -----")
# print(f"Status: {Results.status}")
# print(f"Results array contains {Results.U.size} entries.")
# print(f"Deformed shape: {Results.deformed_shape is not None}")
# print(f"Nodal displacements array shape: {Results.U_n}")
# print(f"Deformed Shape Matrix is : {A}")
# nodes = np.zeros((A.number_of_vertices(), 3), dtype=float)
# print(nodes.shape)

# # Version that minimizes dictionary lookups
# v = A.vertex
# n = A.number_of_vertices()
# for i in range(n):
#     p = v[i]
#     nodes[i] = (p["x"], p["y"], p["z"])
# # --------------------------------------------


# faces = []
# m = A.number_of_faces()

# F = A.face_vertices

# for j in range(m):
#     face = F(j)
#     faces.append([len(face)] + face)
# faces = np.asarray(faces, dtype=np.int64)
# connectivity = np.hstack(faces)


# Pyvista Related functions to be wrapped later
def assign_scalar(mesh, scalar, name):
    """Assign a scalar array to the mesh points."""
    if len(scalar) == mesh.number_of_points:
        mesh.point_data[name] = scalar
    elif len(scalar) == mesh.number_of_cells:
        mesh.cell_data[name] = scalar
    else:
        raise ValueError("Length of scalar array does not match number of points or cells.")
    return None


def mesh_2d_to_pyvista(mesh: Mesh) -> pv.UnstructuredGrid:
    """Convert a Compas Mesh to a PyVista UnstructuredGrid."""
    k = list(mesh.vertex_gkey())
    nodes = np.zeros((mesh.number_of_vertices(), 3), dtype=float)
    for i, key in enumerate(k):
        p = mesh.vertex[key]
        nodes[i] = (p["x"], p["y"], p["z"])

    faces = []
    for j in range(mesh.number_of_faces()):
        face = mesh.face_vertices(j)
        faces.append([len(face)] + face)
    faces = np.asarray(faces, dtype=np.int64)
    connectivity = np.hstack(faces)

    return pv.UnstructuredGrid(connectivity, np.full(mesh.number_of_faces(), pv.CellType.POLYGON), nodes)


# def callback(mesh):
#     g = mesh.picked_mesh
#     print(f"Picked Mesh: {g}")
#     if g is None:
#         return
#     pid = g.find_closest_point(plt.picked_point)
#     u = float(g.point_data["Displacement"][pid])
#     print("pid =", pid, "U =", u)


def mesh_pr2d_to_pyvista(mesh: list[Mesh], scalar) -> pv.UnstructuredGrid:
    """Convert a Compas Mesh to a PyVista UnstructuredGrid."""
    plt = pv.Plotter()
    n = len(mesh)
    for i in range(n):
        face = mesh[i]
        temp = mesh_2d_to_pyvista(face)
        assign_scalar(temp, scalar[i], "Displacement")
        plt.add_mesh(temp, show_edges=True, cmap="jet", opacity=1.0)
    plt.enable_element_picking(show_message=True, left_clicking=True)
    # J = plt.enable_element_picking(mode="edge")
    # print("Selected element edge is :", J)
    return plt


# -------------------------------------------------------------------------

# Process results to get displacement magnitudes

U_n = Results.U_n
U_n = np.reshape(U_n, (len(A), 4, 3))
Ux, Uy = U_n[:, :, 0], U_n[:, :, 1]
U_mag = np.sqrt(Ux**2 + Uy**2)

# -------------------------------------------------------------------------

# Trame server setup and plotting

# pv.OFF_SCREEN = True

# server = get_server()
# state, ctrl = server.state, server.controller

# plt = mesh_pr2d_to_pyvista(A, U_mag)  # Instantiate the plotter.
# # Adds the Mesh and scalar data to the plotter.


# with SinglePageLayout(server) as layout:
#     with layout.content:
#         view = plotter_ui(plt)

# server.start()
# -------------------------------------------------------------------------

# Alternative Pyvista visualization (without trame) ----------------------

# grid = mesh_2d_to_pyvista(A)
# # grid = pv.UnstructuredGrid(connectivity, np.full(A.number_of_faces(), pv.CellType.POLYGON), nodes)
# # grid.plot(show_edges=True)
# plt = pv.Plotter()
# U = Results.U.reshape(grid.number_of_cells, 3)
# Ux, Uy = U[:, 0], U[:, 1]
# U_mag = np.sqrt(Ux**2 + Uy**2)
# Uy_np = np.asarray(Uy, dtype=float)

plt = mesh_pr2d_to_pyvista(A, U_mag)  # Instantiate the plotter.

# assign_scalar(grid, U_mag, "U_mag")
# plt.add_mesh(grid, show_edges=True, cmap="jet", opacity=1.0, show_scalar_bar=False)
# plt.add_scalar_bar(title="Displacement Magnitude", title_font_size=35, label_font_size=20)
# # plt.show_grid()
# # plt.enable_surface_point_picking()
# # plt.add_mesh_clip_plane(grid)
# # plt.add_mesh_slice_spline(grid, n_handles=2)


plt.show()
