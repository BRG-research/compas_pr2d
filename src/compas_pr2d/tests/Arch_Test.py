import compas.geometry as cg
from template.arch import construct_arch
from problem.variables import Problem
from problem.helpers import print_matrix
import numpy as np
import cvxpy as cp
from compas_viewer import Viewer
from compas.colors import Color
import copy
from compas_viewer.scene.tagobject import Tag


# Helper functions
def set_BCs(b, Idx, vector):
    """Sets the boundary conditions in the RHS vector b"""
    b[Idx*2    ] = vector[0]
    b[Idx*2 + 1] = vector[1]
    return b

def set_force_BCs(c, Idx, vector):
    """Set force boundary conditions in the global force vector C"""
    c[Idx*3    ] = vector[0]
    c[Idx*3 + 1] = vector[1]
    c[Idx*3 + 2] = vector[2]
    return c

def deform_polygon_linear(poly, Ux, Uz, th):
    c = poly.centroid()
    cx, cz = c[0], c[2]

    new_pts = []
    for p in range(4):  # or poly.vertices, depending on compas version
        x, z = poly.vertex[p]["x"], poly.vertex[p]["z"]
        x_new = x + Ux - th*(z - cz)
        z_new = z + Uz + th*(x - cx)
        new_pts.append(cg.Point(x_new, 0.0, z_new))

    return cg.Polygon(new_pts)

# Problem parameters -----------------
n = 10
height = 3
span = 10
thickness = 1
g = -9.81 # gravity acceleration
# --------------------------

# Problem Initialization -----------------
Arch_Object = construct_arch(height, span, n, thickness)
Arch_Object_0 = copy.deepcopy(Arch_Object)
The_Problem = Problem.from_polygons(Arch_Object)
# --------------------------

# Extract problem data -----------------
c = The_Problem.C
u = The_Problem.U
intfn = The_Problem.intfn
A_ub = The_Problem.A_ub
A_eq = The_Problem.A_eq

b_ub = np.zeros(A_ub.shape[0])
b_eq = np.zeros(A_eq.shape[0])
# --------------------------
for i in range(n):
    set_force_BCs(c, i, (0, g, 0.0))

# index_BC_x = [0, n-1]
# for i in index_BC_x:
#     set_force_BCs(c, i, (-1.0, 0.0, -5))

#set_force_BCs(c, 2, (0.0, 0.0, 1.0))
#set_force_BCs(c, n-2, (0.0, 0.0, 1.0))

# set_BCs(b_ub, 0, (0.04, 000));
# set_BCs(b_ub, 0, (-0.50, -0.50));
set_BCs(b_eq, 0, (-0.5, -0.5));
# set_BCs(b_eq, n-2, (0.05, 0.05));



# Define and solve the problem -----------------

U = cp.Variable(3*n)
# BC on displacements and rotations of block i
# i * 3 + 0 -> dx
# i * 3 + 1 -> dy
# i * 3 + 2 -> rz

constraints = [
    A_ub @ U >= b_ub,
    A_eq @ U == b_eq,
]

objective = cp.Maximize(c @ U)  # minimize −c·U  (equiv. maximize c·U)
prob = cp.Problem(objective, constraints)

prob.solve(solver=cp.MOSEK, verbose=False)  # or CBC / ECOS

print("Feasible:", prob.status)
U_star = U.value
The_Problem.U = U_star

A_eq_U = A_eq @ U_star
A_ub_U = A_ub @ U_star


U_new = U_star.reshape((n, 3))

# build deformed polygons for viewing
lintl_def = []
for i, poly in enumerate(Arch_Object_0):          # use original undeformed copy
    dx, dz, th = U_new[i]                   # your (Ux, Uz, theta)
    lintl_def.append(deform_polygon_linear(poly, dx, dz, th))


viewer = Viewer()
for idx, polygon in enumerate(lintl_def):
    viewer.scene.add(polygon,facecolor=Color(0.0, 0.0, 1.0), show_points=True, pointcolor=Color(1.0, 0.0, 1.0), linecolor=Color(1.0, 0.5, 1.0), linewidth=1, opacity=0.5)
    # dx, dy, rz = U_new[i]
    # string = f"dx: {dx:.3f} \n dy: {dy:.3f} \n rz: {rz:.3f}"
    # # string = idx
    # t1 = Tag(text=idx, position=polygon.centroid(), color=Color(1.0, 0.0, 0.0))
    # viewer.scene.add(t1)

viewer.show()
