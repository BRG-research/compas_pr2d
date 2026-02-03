# Note::::
# To run this example, you need to have pyvista installed.
# You can install it via pip:
# pip install pyvista

import numpy as np
import pyvista as pv
from compas.datastructures import Mesh

from compas_pr2d import arch
from compas_pr2d.prd import PR2DModel

height = 4
span = 10
n = 5
thickness = 0.5

model = PR2DModel()
arch_geometry = arch.construct_arch(height, span, n, thickness)
model.from_polygons(arch_geometry)
# model.display_mesh(bindex=True, nindex=True)
model.set_boundary_edges(display=False)
disp_data = [
    [4, 0.0, -0.2],
]
# model.set_force()  # for now no loads, just self-weight which is default in the model (i will add as gravity to all blocks for now only)
model.assign_bc(disp_data)
model.solve()
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
    nodes = np.zeros((mesh.number_of_vertices(), 3), dtype=float)
    for i in range(mesh.number_of_vertices()):
        p = mesh.vertex[i]
        nodes[i] = (p["x"], p["y"], p["z"])

    faces = []
    for j in range(mesh.number_of_faces()):
        face = mesh.face_vertices(j)
        faces.append([len(face)] + face)
    faces = np.asarray(faces, dtype=np.int64)
    connectivity = np.hstack(faces)

    return pv.UnstructuredGrid(connectivity, np.full(mesh.number_of_faces(), pv.CellType.POLYGON), nodes)


def mesh_pr2d_to_pyvista(mesh: list[Mesh], scalar) -> pv.UnstructuredGrid:
    """Convert a Compas Mesh to a PyVista UnstructuredGrid."""
    plt = pv.Plotter()
    n = len(mesh)
    for i in range(n):
        face = mesh[i]
        temp = mesh_2d_to_pyvista(face)
        plt.add_mesh(temp, scalars=scalar[i], show_edges=True, cmap="jet", opacity=1.0)
    plt.show()
    return None


U_n = Results.U_n
U_n = np.reshape(U_n, (len(A), 4, 3))
Ux, Uy = U_n[:, :, 0], U_n[:, :, 1]
U_mag = np.sqrt(Ux**2 + Uy**2)
mesh_pr2d_to_pyvista(A, U_mag)


# grid = mesh_2d_to_pyvista(A)
# # grid = pv.UnstructuredGrid(connectivity, np.full(A.number_of_faces(), pv.CellType.POLYGON), nodes)
# # grid.plot(show_edges=True)
# plt = pv.Plotter()
# U = Results.U.reshape(grid.number_of_cells, 3)
# Ux, Uy = U[:, 0], U[:, 1]
# U_mag = np.sqrt(Ux**2 + Uy**2)
# Uy_np = np.asarray(Uy, dtype=float)


# assign_scalar(grid, U_mag, "U_mag")
# plt.add_mesh(grid, show_edges=True, cmap="jet", opacity=1.0, show_scalar_bar=False)
# plt.add_scalar_bar(title="Displacement Magnitude", title_font_size=35, label_font_size=20)
# # plt.show_grid()
# # plt.enable_surface_point_picking()
# # plt.add_mesh_clip_plane(grid)
# # plt.add_mesh_slice_spline(grid, n_handles=2)

# plt.show()
