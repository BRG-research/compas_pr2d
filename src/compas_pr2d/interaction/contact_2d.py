"""Identify interfaces for PRD extended assembly data structures."""

import compas.geometry as cg
from compas.datastructures import Mesh

def equal_points(p1, p2, precision):
    return (
        abs(p1["x"] - p2["x"]) <= precision
        and abs(p1["y"] - p2["y"]) <= precision
        and abs(p1["z"] - p2["z"]) <= precision
    )


def same_edge(a1, b1, a2, b2, precision):
    return (equal_points(a1, a2, precision) and equal_points(b1, b2, precision)) or (
        equal_points(a1, b2, precision) and equal_points(b1, a2, precision)
    )

def contact_2d(polygons: list[cg.Polygon], precision=1e-5):
    """
    Contact Detection for 2D Edges between polygons.

    :param polygons: Polygon list as Compas Geometry objects
    :type polygons: list[cg.Polygon]
    :param precision: Contact Detection Precision
    :type precision: float
    :return: List of interfaces as tuples (block_i, block_j, (edge_i, edge_j))
    :rtype: list[tuple[int, int, tuple[int, int]]]
    """
    interface = []
    edges = []
    for i, arch_1 in enumerate(polygons):
        for j, arch_2 in enumerate(polygons):
            if j <= i:  # avoids i==j and double counting
                continue
            for edge_1 in arch_1.edges_on_boundary():
                a1 = arch_1.vertex[edge_1[0]]
                b1 = arch_1.vertex[edge_1[1]]

                for edge_2 in arch_2.edges_on_boundary():
                    a2 = arch_2.vertex[edge_2[0]]
                    b2 = arch_2.vertex[edge_2[1]]

                    if same_edge(a1, b1, a2, b2, precision):
                        interface.append((i, j, edge_1))
                        #global indexing of the edge
                        N1 = edge_1[0] + i * 4
                        N2 = edge_1[1] + i * 4
                        N3 = edge_2[0] + j * 4
                        N4 = edge_2[1] + j * 4
                        edges.append((N1, N2))
                        edges.append((N3, N4))

    return interface, edges

def common_edges(Mesh: Mesh, precision=1e-5):
    """
    Identify common edges between polygons.

    :param polygons: Polygon list as Compas Geometry objects
    :type polygons: list[cg.Polygon]
    :param precision: Contact Detection Precision
    :type precision: float
    :return: List of common edges as tuples (block_i, block_j, (edge_i, edge_j))
    :rtype: list[tuple[int, int, tuple[int, int]]]
    """
    E = list(Mesh.edges())
    E_Contact = []
    for edge in E:
        i,j = Mesh.edge_faces(edge)
        if i is not None and j is not None:
            E_Contact.append((i,j,edge))
        
    return E_Contact