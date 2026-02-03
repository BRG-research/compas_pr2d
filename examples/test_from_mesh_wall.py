from compas_pr2d import PR2DModel
from compas_pr2d import wall

print("Constructing the model ...")
model = PR2DModel()
print("Constructing the wall geometry ...")

nl = 10
nh = 4
length = 10
height = 6

wall_geometry = wall.construct_wall(length, height, nl, nh)
print("Feeding the geometry to the model ...")

model.from_polygons(wall_geometry)
# model.display_mesh(bindex=True, nindex=True)  # Keep only for when boundary edges needs to be chosen individually

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

print("Assembling Matrices:")

model.set_boundary_edges(display=True, boundary_edges=boundary_edges)
disp_data = [
    [66, (0.3, 0.25), (0.0, 0.00)],
    [67, (0.25, 0.2), (0.0, 0.00)],
    [68, (0.2, 0.15), (0.0, 0.00)],
    [69, (0.15, 0.1), (0.0, 0.00)],
    [70, (0.1, 0.05), (0.0, 0.00)],
    [71, (0.05, 0.0), (0.0, 0.00)],
]  # [Interface ID, normal displacement, tangential displacement]

# disp_data = [
#     [66, (0.0, 0.0), (0.05, 0.005)],
#     [67, (0.0, 0.0), (0.05, 0.005)],
#     [68, (0.0, 0.0), (0.05, 0.005)],
#     [69, (0.0, 0.0), (-0.05, -0.005)],
#     [70, (0.0, 0.0), (-0.05, -0.005)],
#     [71, (0.0, 0.0), (-0.05, -0.005)],
# ]
# model.set_force()  # for now no loads, just self-weight which is default in the model (i will add as gravity to all blocks for now only)
print("Assigning BCs to RHS vectors...")
model.assign_bc(disp_data)
# print("Solving the system...")
model.solve()

model.display_results()  # postprocesses and displays results
