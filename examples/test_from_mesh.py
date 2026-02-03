from compas_pr2d import arch
from compas_pr2d.prd import PR2DModel

height = 4
span = 10
n = 5
thickness = 0.5

print("Constructing the model ...")
model = PR2DModel()
print("Constructing the arch geometry ...")
arch_geometry = arch.construct_arch(height, span, n, thickness)
print("Feeding the geometry to the model ...")

model.from_polygons(arch_geometry)
# model.display_mesh(bindex=True, nindex=True)

print("Assembling Matrices:")

model.set_boundary_edges(display=True)

disp_data = [
    [4, 0.0, -0.2],
]  # [Interface ID, normal displacement, tangential displacement]


# model.set_force()  # for now no loads, just self-weight which is default in the model (i will add as gravity to all blocks for now only)
print("Assigning BCs to RHS vectors...")
model.assign_bc(disp_data)
# print("Solving the system...")
model.solve()
# model.display_results()  # postprocesses and displays results
