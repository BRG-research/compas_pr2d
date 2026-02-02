from math import radians

import compas.geometry as cg
from compas.datastructures import Mesh


def construct_lintles(height, span, n):
    thickness = span / n

    blocks = []
    for i in range(n):
        a = [-span / 2 + i * thickness, 0.0, height]
        b = [-span / 2 + i * thickness, 0.0, 0.0]
        c = [-span / 2 + (i + 1) * thickness, 0.0, 0.0]
        d = [-span / 2 + (i + 1) * thickness, 0.0, height]
        mesh = Mesh.from_vertices_and_faces([a, b, c, d], [[0, 1, 2, 3]])
        blocks.append(mesh)

    return blocks
