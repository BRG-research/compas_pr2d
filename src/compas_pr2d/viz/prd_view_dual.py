import compas.geometry as cg
from compas_viewer import Viewer


def weighted_point_on_segment(p0, p1, w0, w1):
    wsum = w0 + w1
    if wsum == 0:
        return cg.Point((p0.x + p1.x) / 2, (p0.y + p1.y) / 2, (p0.z + p1.z) / 2)
    else:
        return cg.Point(
            (w0 * p0.x + w1 * p1.x) / wsum,
            (w0 * p0.y + w1 * p1.y) / wsum,
            (w0 * p0.z + w1 * p1.z) / wsum,
        )


def show_dual_results(model, fn_opt, ft_opt, scale=0.01):
    edges = model.all_contacts
    nodes = model.n
    deformed = model.results.deformed_shape

    viewer = Viewer()
    for shape in deformed:
        viewer.scene.add(shape)

    for eid, e in enumerate(edges):
        if len(e) == 4:
            i, j, k, (u, v) = e
        elif len(e) == 3:
            i, k, (u, v) = e
        else:
            continue

        p0 = cg.Point(nodes[u][0], nodes[u][1], nodes[u][2])
        p1 = cg.Point(nodes[v][0], nodes[v][1], nodes[v][2])

        line = cg.Line(p0, p1)

        # ---- get endpoint weights (EXAMPLE)
        # Replace these with your two endpoint values once you have them.
        fn0 = float(abs(fn_opt[eid]))  # end at node u
        fn1 = float(abs(fn_opt[eid]))  # end at node v

        pc = weighted_point_on_segment(p0, p1, fn0, fn1)

        # Normal (assuming section is XZ and y is out-of-plane)
        t = cg.Vector(*line.vector)
        t.y = 0.0
        if t.length == 0:
            raise ValueError("Zero-length edge found.")
        t.unitize()

        n = cg.Vector(-t.z, 0.0, t.x)
        if n.length == 0:
            raise ValueError("Zero-length normal vector found.")
        n.unitize()

        # Magnitude (use average or resultant; here average)
        fn_mag = 0.5 * (fn0 + fn1)

        tip = cg.Point(pc.x, pc.y, pc.z)
        tip.translate(n * (scale * fn_mag))

        viewer.scene.add(cg.Line(pc, tip))  # <-- NO anchor here

    viewer.show()
