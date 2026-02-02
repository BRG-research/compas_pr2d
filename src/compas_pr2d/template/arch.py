from math import radians

import compas.geometry as cg
from compas.datastructures import Mesh


def construct_arch(rise, span, n, thickness):
    radius = rise / 2 + span**2 / (8 * rise)
    # base = [0.0, 0.0, 0.0]
    top = [0.0, 0.0, rise]
    left = [-span / 2, 0.0, 0.0]
    center = [0.0, 0.0, rise - radius]
    vector = cg.subtract_vectors(left, center)
    springing = cg.angle_vectors(vector, [-1.0, 0.0, 0.0])
    sector = radians(180) - 2 * springing
    angle = sector / n

    a = top
    b = cg.add_vectors(top, [0, 0, 0])
    c = cg.add_vectors(top, [0, 0, thickness])
    d = cg.add_vectors(top, [0, 0, thickness])

    R = cg.Rotation.from_axis_and_angle([0, 1.0, 0], 0.5 * sector, center)
    bottom = cg.transform_points([a, b, c, d], R)

    blocks = []
    for _ in range(n):
        R = cg.Rotation.from_axis_and_angle([0, 1.0, 0], -angle, center)
        top = cg.transform_points(bottom, R)
        vertices = bottom[0], bottom[2], top[2], top[0]
        faces = [
            [0, 1, 2, 3],
        ]
        mesh = Mesh.from_vertices_and_faces(vertices, faces)

        blocks.append(mesh)
        bottom = top

    return blocks
