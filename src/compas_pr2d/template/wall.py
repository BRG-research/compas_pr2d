from math import radians

import compas.geometry as cg
from compas.datastructures import Mesh


def construct_wall(length, height, nl, nh):
    l_block = length / nl
    h_block = height / nh
    blocks = []
    for i in range(nh):
        h = (i) * h_block
        for j in range(nl):
            a = [j * l_block, 0.0, h]
            b = [j * l_block, 0.0, h + h_block]
            c = [(j + 1) * l_block, 0.0, h + h_block]
            d = [(j + 1) * l_block, 0.0, h]
            mesh = Mesh.from_vertices_and_faces([a, b, c, d], [[0, 1, 2, 3]])
            blocks.append(mesh)

    return blocks
