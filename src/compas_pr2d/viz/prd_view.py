import compas.geometry as cg
from compas_viewer import Viewer
from compas.colors import Color
from compas_viewer.scene import Tag
from compas_viewer.scene import Collection


def show_mesh(mesh, bindex: bool = False, nindex: bool = False, n=None, e=None) -> None:
    """Visualize polygons with their edges and normals using Compas Viewer.
    Displays global node indices or block indices if specified.
    """
    viewer = Viewer()
    # viewer.scene.clear()
    viewer.scene.add(
        mesh,
        show_vertices=True,
        show_lines=True,
        pointcolor=Color(1.0, 0.0, 0.0),
        opacity=0.25,
    )
    # print(f"Coorindates: {n}")
    # nindices = []
    if nindex:
        group_1 = viewer.scene.add_group(name="Node Indices")

        for x in range(len(n)):
            t = Tag(x.__str__(), position=n[x], height=40)
            group_1.add(t)
            # nindices.append(t)

        # viewer.scene.add(Collection(nindices), name = "Node Indices")

    # eindices = []
    if bindex:
        group_2 = viewer.scene.add_group(name="Interface Indices")

        #     # Build undirected edge -> label id
        for i, (u, v) in enumerate(mesh.edges()):  # directed edges
            cg.centroid_points([n[u], n[v]])
            t = Tag(
                i.__str__(),
                position=cg.centroid_points([n[u], n[v]]),
                height=40,
                color=Color(1, 0, 0.2),
            )
            group_2.add(t)
            # eindices.append(t)
        # viewer.scene.add(Collection(eindices), name = "Edge Indices")

    viewer.show()
    # raise Exception(
    #     "Please assign boundary edges and displacement BCs in the test file. \n Then comment this line to proceed."
    # )


def show_mesh_with_contact(mesh, bindex: bool = True, n=None, e=None) -> None:
    """Visualize polygons with their edges and normals using Compas Viewer.
    Displays global node indices or block indices if specified.
    """
    viewer = Viewer()
    # viewer.scene.clear()
    viewer.scene.add(
        mesh,
        show_vertices=True,
        show_lines=True,
        pointcolor=Color(1.0, 0.0, 0.0),
        opacity=0.25,
    )

    #     # Build undirected edge -> label id
    for edg in e:  # directed edges
        if len(edg) == 3:
            _, i, (u, v) = edg
        elif len(edg) == 4:
            _, _, i, (u, v) = edg
        else:
            raise Exception(f"Edge format {edg} not recognized for visualization.")

        cg.centroid_points([n[u], n[v]])
        t = Tag(
            i.__str__(),
            position=cg.centroid_points([n[u], n[v]]),
            height=40,
            color=Color(1, 0, 0.2),
        )
        viewer.scene.add(t)
    viewer.show()
    # viewer.show()
    # raise Exception(
    #     "Please assign boundary conditions at the shown interfaces. \n Then comment this line to proceed."
    # )
