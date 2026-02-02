from prd import PRDModel
from template.arch import construct_arch

print("Constructing the model:")
model = PRDModel()
print("Constructing the arch geometry:")
arch_geometry = construct_arch(4, 10, 12, 0.5)
print("Feeding the geometry to the model:")
model.from_polygons(arch_geometry)
# model.display_polygons(bindex=False, nindex=True)

model_edges = [
    (0, 1),
    (46, 47),
]  # Example edges defined after looking at the arch structure

disp_data = [
    [0, -0.05, 0],
    # [5, 0.1, 0.0],
]  # [Block ID, normal displacement, tangential displacement] (I will define a parser for this in model to call the function)

print("Assembling Matrices:")

model.set_model_edges(model_edges)
model.set_displacement_bc(disp_data)
# model.set_force()  # for now no loads, just self-weight which is default in the model (i will add as gravity to all blocks for now only)
print("Assigning BCs to RHS vectors...")
model.assign_bc()
print("Solving the system...")
model.solve()
model.display_results()  # postprocesses and displays results
