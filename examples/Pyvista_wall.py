# Note::::
# To run this example, you need to have pyvista installed.
# You can install it via pip:
# pip install pyvista

import numpy as np
import pyvista as pv
from compas.datastructures import Mesh

from compas_pr2d import PR2DModel
from compas_pr2d import wall

# Instantiate the model
model = PR2DModel()

# Construct wall geometry
nl = 10
nh = 4
length = 10
height = 6

wall_geometry = wall.construct_wall(length, height, nl, nh)
# ------------------------------------------------------------------------

# Feed the geometry to the model
model.from_polygons(wall_geometry)
# model.display_mesh(bindex=True, nindex=True)  # Keep only for when boundary edges needs to be chosen individually

# Define boundary edges manually (optional)
boundary_edges = [
    1,
    7,
    11,
    15,
    19,
    23,
    27,
    31,
    35,
    39,
]  # Optional list of boundary edge IDs.
# Otherwise, boundary edges will be automatically detected.
# But works only for edges with one block.

# ------------------------------------------------------------------------

model.set_boundary_edges(display=False, boundary_edges=boundary_edges)

# Define displacement boundary conditions --------------------------------
disp_data = [
    [66, (0.3, 0.25), (0.0, 0.00)],
    [67, (0.25, 0.2), (0.0, 0.00)],
    [68, (0.2, 0.15), (0.0, 0.00)],
    [69, (0.15, 0.1), (0.0, 0.00)],
    [70, (0.1, 0.05), (0.0, 0.00)],
    [71, (0.05, 0.0), (0.0, 0.00)],
]  # [Interface ID, normal displacement, tangential displacement]

# model.set_force()  # for now no loads, just self-weight which is default in the model (i will add as gravity to all blocks for now only)
model.assign_bc(disp_data)

# -------------------------------------------------------------------------
# Solve the model ---------------------------------------------------------

model.solve()
# model.display_results()  # postprocesses and displays results
# -------------------------------------------------------------------------


# Visualize results using PyVista ------------------------------------------
A = model.mesh
Results = model.results

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


grid = mesh_2d_to_pyvista(A)
# grid = pv.UnstructuredGrid(connectivity, np.full(A.number_of_faces(), pv.CellType.POLYGON), nodes)
# grid.plot(show_edges=True)
plt = pv.Plotter()
U = Results.U.reshape(grid.number_of_cells, 3)
Ux, Uy = U[:, 0], U[:, 1]
U_mag = np.sqrt(Ux**2 + Uy**2)
Uy_np = np.asarray(Uy, dtype=float)


assign_scalar(grid, U_mag, "U_mag")
plt.add_mesh(grid, show_edges=True, cmap="jet", opacity=1.0, show_scalar_bar=False)
plt.add_scalar_bar(title="Displacement Magnitude", title_font_size=35, label_font_size=20)
# plt.show_grid()
# plt.enable_surface_point_picking()
# plt.add_mesh_clip_plane(grid)
# plt.add_mesh_slice_spline(grid, n_handles=2)

plt.show()
