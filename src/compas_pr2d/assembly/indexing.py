from typing import List
import compas.geometry as cg
from compas.datastructures import Mesh


def mesh_polygon(polygons: List[Mesh]):
    """
    converts the list of compas polygons to a compas mesh
    :polygons: List[cg.Polygon] -- list of compas polygons
    :return: Mesh -- compas mesh
    """
    print(f"Polygon points are: {polygons[0].face_points(0)}")
    polygons_vertices = [[[p[0], p[1], p[2]] for p in poly.face_points(0)] for poly in polygons]
    # Instead of polygons.to_vertices_and_faces(), get only vertices

    meshed = Mesh.from_polygons(polygons_vertices)

    return meshed


def mesh_to_vertices_edges(mesh: Mesh):
    """
    extracts the vertices and edges from a compas mesh
    :mesh: Mesh -- compas mesh
    :return: List, List -- list of vertices and list of edges
    """
    vertices = []
    for idx in range(mesh.number_of_vertices()):
        x, y, z = mesh.vertex_coordinates(idx)
        vertices.append(cg.Point(x, y, z))

    edges = list(mesh.edges())
    return vertices, edges


def common_edges(Mesh: Mesh, assigned_boundary_edges=None, precision=1e-5):
    """
    Identify common edges between polygons.

    :param polygons: Polygon list as Compas Geometry objects
    :type polygons: list[cg.Polygon]
    :param precision: Contact Detection Precision
    :type precision: float
    :return: List of common edges as tuples (block_i, block_j, intfnumber, (edge_i, edge_j))
    :rtype: list[tuple[int, int, int, tuple[int, int]]]
    """
    E = list(Mesh.edges())

    boundary_edges = []
    E_Contact = []
    k = 0
    for edge in E:
        i, j = Mesh.edge_faces(edge)
        if i is not None and j is not None:
            E_Contact.append((i, j, k, edge))
            k += 1
    if assigned_boundary_edges is None:
        boundary_edges = find_boundary_edges(Mesh, k)
    else:
        for b in assigned_boundary_edges:
            e = E[b]
            u, v = e
            i, j = Mesh.edge_faces(e)
            if i is None:
                boundary_edges.append((j, k, (u, v)))
                k += 1
            elif j is None:
                boundary_edges.append((i, k, (u, v)))
                k += 1
            else:
                raise Exception("Boundary edge specified is not a boundary edge.")

    return E_Contact, boundary_edges


def find_boundary_edges(mesh: Mesh, k):
    """
    Identify boundary edges in the mesh.

    :param mesh: Input Mesh
    :type mesh: Mesh
    :return: List of boundary edges as tuples (edge_index, (vertex1, vertex2))
    :rtype: list[tuple[int, tuple[int, int]]]
    """
    E = list(mesh.edges())
    boundary_edges = []
    for edge in E:
        if mesh.is_edge_on_boundary(edge):
            for p in edge:
                if len(mesh.vertex_faces(edge[0])) == 1 and len(mesh.vertex_faces(edge[1])) == 1:
                    face = mesh.vertex_faces(edge[1])[0]
                    boundary_edges.append((face, k, edge))
                    k += 1

                    break

    return boundary_edges
