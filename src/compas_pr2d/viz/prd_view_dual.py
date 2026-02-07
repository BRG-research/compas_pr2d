import compas.geometry as cg
from compas.colors import Color
from compas_viewer import Viewer
from compas_viewer.scene import Tag


def weighted_point_on_segment(p0, p1, w0, w1):
    wsum = w0 + w1
    if wsum == 0:
        return cg.Point(0.5 * (p0.x + p1.x), 0.5 * (p0.y + p1.y), 0.5 * (p0.z + p1.z))
    return cg.Point(
        (w0 * p0.x + w1 * p1.x) / wsum,
        (w0 * p0.y + w1 * p1.y) / wsum,
        (w0 * p0.z + w1 * p1.z) / wsum,
    )


def show_dual_results(model, fn_opt, ft_opt, scale=0.01):
    edges = model.all_contacts
    nodes = model.n

    # Later to add plotting forces on deformed shape, for now just the original geometry with forces as tags and colors
    deformed = model.results.deformed_shape
    # for shape in deformed:
    #     viewer.scene.add(shape)

    points_t = []
    viewer = Viewer()

    viewer.scene.add(model.mesh, opacity=0.5)

    scale = scale
    print(len(edges))
    for e in edges:
        if len(e) == 4:
            _, _, k, (u, v) = e
        elif len(e) == 3:
            _, k, (u, v) = e
        else:
            continue

        A = cg.Point(*nodes[u])
        B = cg.Point(*nodes[v])
        line = cg.Line(A, B)

        # Get the normal force values at the endpoints of the edge

        fnA = float(fn_opt[2 * k])  # end at node A
        fnB = float(fn_opt[2 * k + 1])  # end at node B

        # Weighted application point
        wA, wB = abs(fnA), abs(fnB)
        N = fnA + fnB
        pc = weighted_point_on_segment(A, B, wA, wB)

        t = cg.Vector(*line.vector)
        t.y = 0.0
        if t.length == 0:
            continue
        t.unitize()

        n = cg.Vector(-t.z, 0.0, t.x)
        if n.length == 0:
            continue
        n.unitize()

        # Magnitude (use average or resultant; here average)
        # tip = cg.Point(pc.x, pc.y, pc.z)
        # tip.translate(n * (scale * N))
        points_t.append([pc, N])
        # viewer.scene.add(cg.Line(pc, tip))   # <-- NO anchor here
        t = Tag(k.__str__(), position=pc)
        # viewer.scene.add(t)

    points, N = zip(*points_t)
    points = list(points)
    polyline = cg.Polyline(points)

    for point, n in zip(points, N):
        viewer.scene.add(point, vertexcolor=Color.from_number(n), pointsize=25)
    # viewer.scene.add(polyline)
    viewer.show()
