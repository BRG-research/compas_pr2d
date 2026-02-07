from compas_pr2d import PR2DModel
from compas_pr2d import lintles

height = 4
span = 10
n = 10
thickness = 0.5


print("Constructing the model ...")
model = PR2DModel()
print("Constructing the lintles geometry ...")
lintles_geometry = lintles.construct_lintles(height, span, n)
print("Feeding the geometry to the model ...")

model.from_polygons(lintles_geometry)
# model.display_mesh(bindex=True, nindex=True)  # Keep only for debugging

# boundary_edges = [0,10] # Optional list of boundary edge IDs.
# Otherwise, boundary edges will be automatically detected.
# But works only for edges with one block.

print("Assembling Matrices:")

model.set_boundary_edges(display=True)

disp_data = [
    [10, (0.2, 0.1), (0.0, 0.00)],
]  # [Interface ID, normal displacement, tangential displacement]

# model.set_force()  # for now no loads, just self-weight which is default in the model (i will add as gravity to all blocks for now only)
print("Assigning BCs to RHS vectors...")
model.assign_bc(disp_data)
# print("Solving the system...")
model.solve()
model.display_results()  # postprocesses and displays results
