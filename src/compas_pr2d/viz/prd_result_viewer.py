import compas.geometry as cg
from compas_viewer import Viewer
from compas.colors import Color
from compas_viewer.scene.tagobject import Tag
import numpy as np
from compas.datastructures import Mesh


def show_results(polygons, results, scale=1.0):
    """Visualize polygons with their edges and normals using Compas Viewer.
    Displays global node indices or block indices if specified.
    """
    n = len(list(polygons.faces()))
    U_solution = np.asarray(results.U)
    print(U_solution)
    U_new = U_solution.reshape((n, 3))

    # build deformed polygons for viewing
    polygons_def = []
    for i in range(n):  # use original undeformed copy
        dx, dz, th = U_new[i]  # your (Ux, Uz, theta)
        polygons_def.append(deform_polygon_linear(polygons, dx, dz, th, i))

    viewer = Viewer()
    for idx, polygon in enumerate(polygons_def):
        polygon.scale(scale)
        viewer.scene.add(
            polygon,
            facecolor=Color(0.0, 0.0, 1.0),
            show_points=True,
            pointcolor=Color(1.0, 0.0, 1.0),
            linecolor=Color(1.0, 0.5, 1.0),
            linewidth=1,
            opacity=0.5,
        )
    viewer.show()


def deform_polygon_linear(poly, Ux, Uz, th, face_index):
    c = poly.face_centroid(face_index)
    cx, cz = c[0], c[2]
    U_nodes = []
    new_pts = []
    for p in poly.face_vertices(face_index):  #
        x, z = poly.vertex[p]["x"], poly.vertex[p]["z"]
        x_new = x + Ux - th * (z - cz)
        z_new = z + Uz + th * (x - cx)
        new_pts.append(cg.Point(x_new, 0.0, z_new))
        U_nodes.append((x_new - x, z_new - z))

    return cg.Polygon(new_pts)
