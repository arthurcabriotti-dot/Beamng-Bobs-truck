"""Tiny procedural mesh library: primitives + Collada (.dae) and glTF (.glb) writers.

Coordinates follow the BeamNG vehicle convention used by the jbeam:
  +X = vehicle left, +Y = rearward, +Z = up, metres.
"""
import json
import math
import struct


# ---------------------------------------------------------------- vector math
def add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def length(a): return math.sqrt(dot(a, a))
def lerp(a, b, t): return add(a, mul(sub(b, a), t))


def norm(a):
    l = length(a)
    return (0.0, 0.0, 1.0) if l < 1e-12 else mul(a, 1.0 / l)


def centroid(pts):
    n = len(pts)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n)


def mirror_x(p): return (-p[0], p[1], p[2])


# ---------------------------------------------------------------- 2D helpers
def poly_area2(poly):
    a = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        a += x0 * y1 - x1 * y0
    return a * 0.5


def triangulate(poly):
    """Ear clipping for a simple polygon (list of 2D points). Returns index triples, CCW."""
    idx = list(range(len(poly)))
    if poly_area2(poly) < 0:
        idx.reverse()

    def is_ear(i0, i1, i2, rest):
        ax, ay = poly[i0]; bx, by = poly[i1]; cx, cy = poly[i2]
        if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) <= 1e-12:
            return False
        for j in rest:
            px, py = poly[j]
            d1 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            d2 = (cx - bx) * (py - by) - (cy - by) * (px - bx)
            d3 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
            if d1 >= -1e-12 and d2 >= -1e-12 and d3 >= -1e-12:
                return False
        return True

    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[(k - 1) % n], idx[k], idx[(k + 1) % n]
            rest = [j for j in idx if j not in (i0, i1, i2)]
            if is_ear(i0, i1, i2, rest):
                tris.append((i0, i1, i2))
                idx.pop(k)
                break
        else:
            break  # degenerate; bail out with what we have
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


# ---------------------------------------------------------------- mesh object
class MeshObject:
    """One named object in the output file. Triangles are grouped by material."""

    def __init__(self, name):
        self.name = name
        self.groups = {}  # material -> list of (p0,p1,p2,n0,n1,n2,uv0,uv1,uv2)

    # -- low level
    def tri(self, mat, p0, p1, p2, uv=None, n=None):
        fn = norm(cross(sub(p1, p0), sub(p2, p0)))
        if length(cross(sub(p1, p0), sub(p2, p0))) < 1e-14:
            return
        ns = n if n is not None else (fn, fn, fn)
        if uv is None:
            uv = (auto_uv(p0, fn), auto_uv(p1, fn), auto_uv(p2, fn))
        self.groups.setdefault(mat, []).append((p0, p1, p2, ns[0], ns[1], ns[2], uv[0], uv[1], uv[2]))

    def quad(self, mat, a, b, c, d, uv=None, n=None):
        """a,b,c,d counter-clockwise when seen from the front."""
        if uv is None:
            self.tri(mat, a, b, c, n=None if n is None else (n[0], n[1], n[2]))
            self.tri(mat, a, c, d, n=None if n is None else (n[0], n[2], n[3]))
        else:
            self.tri(mat, a, b, c, uv=(uv[0], uv[1], uv[2]), n=None if n is None else (n[0], n[1], n[2]))
            self.tri(mat, a, c, d, uv=(uv[0], uv[2], uv[3]), n=None if n is None else (n[0], n[2], n[3]))

    def oriented_quad(self, mat, a, b, c, d, outward, uv=None):
        """Quad whose winding is flipped if needed so its normal faces `outward`."""
        fn = cross(sub(b, a), sub(c, a))
        if dot(fn, outward) < 0:
            a, b, c, d = d, c, b, a
            if uv is not None:
                uv = (uv[3], uv[2], uv[1], uv[0])
        self.quad(mat, a, b, c, d, uv=uv)

    def merge(self, other):
        for m, t in other.groups.items():
            self.groups.setdefault(m, []).extend(t)

    def transformed(self, fn, flip=False, name=None):
        o = MeshObject(name or self.name)
        for m, tris in self.groups.items():
            out = o.groups.setdefault(m, [])
            for t in tris:
                p = [fn(t[0]), fn(t[1]), fn(t[2])]
                nn = [norm(sub(fn(add(t[i], t[3 + i])), fn(t[i]))) for i in range(3)]
                uv = [t[6], t[7], t[8]]
                if flip:
                    p = [p[0], p[2], p[1]]
                    nn = [nn[0], nn[2], nn[1]]
                    uv = [uv[0], uv[2], uv[1]]
                out.append((p[0], p[1], p[2], nn[0], nn[1], nn[2], uv[0], uv[1], uv[2]))
        return o

    def mirrored(self, name=None):
        return self.transformed(mirror_x, flip=True, name=name)

    def tri_count(self):
        return sum(len(t) for t in self.groups.values())

    # -- primitives
    def box(self, mat, mn, mx, skip=()):
        x0, y0, z0 = mn
        x1, y1, z1 = mx
        c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        faces = {
            "-z": (c[0], c[3], c[2], c[1]), "+z": (c[4], c[5], c[6], c[7]),
            "-y": (c[0], c[1], c[5], c[4]), "+y": (c[2], c[3], c[7], c[6]),
            "-x": (c[3], c[0], c[4], c[7]), "+x": (c[1], c[2], c[6], c[5]),
        }
        for k, f in faces.items():
            if k not in skip:
                self.quad(mat, *f)

    def hexa(self, mat, p, skip=()):
        """General convex hexahedron. p[i][j][k], i=x side (0/1), j=y side, k=z side."""
        cen = centroid([p[i][j][k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        faces = {
            "-x": [p[0][0][0], p[0][1][0], p[0][1][1], p[0][0][1]],
            "+x": [p[1][0][0], p[1][1][0], p[1][1][1], p[1][0][1]],
            "-y": [p[0][0][0], p[1][0][0], p[1][0][1], p[0][0][1]],
            "+y": [p[0][1][0], p[1][1][0], p[1][1][1], p[0][1][1]],
            "-z": [p[0][0][0], p[1][0][0], p[1][1][0], p[0][1][0]],
            "+z": [p[0][0][1], p[1][0][1], p[1][1][1], p[0][1][1]],
        }
        for k, f in faces.items():
            if k in skip:
                continue
            self.oriented_quad(mat, *f, outward=sub(centroid(f), cen))

    def extrude_x(self, mat, poly, x0, x1, cap_mat=None, side_mat=None, skip_caps=False):
        """Extrude a (y,z) polygon between x0 and x1. side_mat may be a function (a, b) -> material|None
        picking the material of the side face swept by polygon edge a->b."""
        self._extrude(mat, poly, x0, x1, lambda u, v, w: (w, u, v), (1, 0, 0), cap_mat, side_mat, skip_caps)

    def extrude_y(self, mat, poly, y0, y1, cap_mat=None, side_mat=None, skip_caps=False):
        """Extrude an (x,z) polygon between y0 and y1."""
        self._extrude(mat, poly, y0, y1, lambda u, v, w: (u, w, v), (0, 1, 0), cap_mat, side_mat, skip_caps)

    def extrude_z(self, mat, poly, z0, z1, cap_mat=None, side_mat=None, skip_caps=False):
        """Extrude an (x,y) polygon between z0 and z1."""
        self._extrude(mat, poly, z0, z1, lambda u, v, w: (u, v, w), (0, 0, 1), cap_mat, side_mat, skip_caps)

    def _extrude(self, mat, poly, w0, w1, emb, axis, cap_mat, side_mat, skip_caps):
        if poly_area2(poly) < 0:
            poly = list(reversed(poly))
        cap_mat = cap_mat or mat
        side_fn = side_mat if callable(side_mat) else (lambda a, b: side_mat)
        if not skip_caps:
            for (i, j, k) in triangulate(poly):
                a, b, c = poly[i], poly[j], poly[k]
                top = [emb(*a, w1), emb(*b, w1), emb(*c, w1)]
                bot = [emb(*a, w0), emb(*b, w0), emb(*c, w0)]
                self._tri_facing(cap_mat, top, axis)
                self._tri_facing(cap_mat, bot, mul(axis, -1))
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            du, dv = b[0] - a[0], b[1] - a[1]
            out2 = (dv, -du)  # right of edge for CCW polygon = outward
            outward = emb(out2[0], out2[1], 0.0)
            q = [emb(*a, w0), emb(*b, w0), emb(*b, w1), emb(*a, w1)]
            self.oriented_quad(side_fn(a, b) or mat, *q, outward=outward)

    def _tri_facing(self, mat, t, want):
        fn = cross(sub(t[1], t[0]), sub(t[2], t[0]))
        if dot(fn, want) < 0:
            t = [t[0], t[2], t[1]]
        self.tri(mat, *t)

    def lathe(self, mat, origin, axis, profile, segs=32, uv_repeat=1.0, smooth=True, a0=0.0, a1=2 * math.pi):
        """Revolve profile [(radius, along_axis), ...] around axis through origin.
        Trace the profile so the outside of the surface is on its right when radius points
        up and the axis points right (e.g. inner bead -> up sidewall -> across tread -> down)."""
        axis = norm(axis)
        tmp = (0, 0, 1) if abs(axis[2]) < 0.9 else (1, 0, 0)
        u = norm(cross(axis, tmp))
        v = cross(axis, u)
        full = abs((a1 - a0) - 2 * math.pi) < 1e-9
        nseg = segs
        ang = [a0 + (a1 - a0) * s / nseg for s in range(nseg + 1)]
        # per profile-segment 2D normal (radial, axial) chosen to point "outside" of the swept
        # shape: rotate the segment direction by -90deg.
        prof = profile
        seglen = [math.hypot(prof[i + 1][0] - prof[i][0], prof[i + 1][1] - prof[i][1]) for i in range(len(prof) - 1)]
        tot = sum(seglen) or 1.0
        vcoord = [0.0]
        for s in seglen:
            vcoord.append(vcoord[-1] + s / tot)

        def pt(r, t, a):
            ca, sa = math.cos(a), math.sin(a)
            return add(add(origin, mul(axis, t)), add(mul(u, r * ca), mul(v, r * sa)))

        def dirv(nr, nt, a):
            ca, sa = math.cos(a), math.sin(a)
            return norm(add(mul(axis, nt), add(mul(u, nr * ca), mul(v, nr * sa))))

        for i in range(len(prof) - 1):
            (r0, t0), (r1, t1) = prof[i], prof[i + 1]
            dr, dt = r1 - r0, t1 - t0
            # normal = (dt, -dr) in (radial, axial): profile traced so outside is on its right
            nr, nt = dt, -dr
            for s in range(nseg):
                aa, ab = ang[s], ang[s + 1]
                p00, p01 = pt(r0, t0, aa), pt(r0, t0, ab)
                p11, p10 = pt(r1, t1, ab), pt(r1, t1, aa)
                uva = (ang[s] / (2 * math.pi) * uv_repeat, vcoord[i])
                uvb = (ang[s + 1] / (2 * math.pi) * uv_repeat, vcoord[i])
                uvc = (ang[s + 1] / (2 * math.pi) * uv_repeat, vcoord[i + 1])
                uvd = (ang[s] / (2 * math.pi) * uv_repeat, vcoord[i + 1])
                if smooth:
                    na, nb = dirv(nr, nt, aa), dirv(nr, nt, ab)
                    ns = (na, nb, nb, na)
                else:
                    ns = None
                quad = [p00, p01, p11, p10]
                uvq = (uva, uvb, uvc, uvd)
                want = dirv(nr, nt, (aa + ab) / 2)
                fn = cross(sub(quad[1], quad[0]), sub(quad[2], quad[0]))
                if length(fn) < 1e-14:
                    fn = cross(sub(quad[2], quad[0]), sub(quad[3], quad[0]))
                if dot(fn, want) < 0:
                    quad = quad[::-1]
                    uvq = uvq[::-1]
                    if ns:
                        ns = ns[::-1]
                self.quad(mat, *quad, uv=uvq, n=ns)

    def cylinder(self, mat, p0, p1, r, segs=12, caps=True, r1=None):
        r1 = r if r1 is None else r1
        ax = sub(p1, p0)
        L = length(ax)
        prof = [(r, 0.0), (r1, L)]
        if caps:
            prof = [(0.0, 0.0)] + prof + [(0.0, L)]
        self.lathe(mat, p0, ax, prof, segs=segs, smooth=True)

    def torus(self, mat, center, axis, R, r, segs=32, rsegs=10):
        prof = []
        for i in range(rsegs + 1):
            a = 2 * math.pi * i / rsegs
            prof.append((R + r * math.cos(a), r * math.sin(a)))
        self.lathe(mat, center, axis, prof, segs=segs)


def auto_uv(p, n):
    ax, ay, az = abs(n[0]), abs(n[1]), abs(n[2])
    if ax >= ay and ax >= az:
        return (p[1], p[2])
    if ay >= az:
        return (p[0], p[2])
    return (p[0], p[1])


def bilerp(q, u, v):
    """q = (a,b,c,d) quad corners (a->b is u, a->d is v)."""
    return lerp(lerp(q[0], q[1], u), lerp(q[3], q[2], u), v)


# ---------------------------------------------------------------- Collada writer
def write_dae(path, objects, materials):
    """objects: list of MeshObject. materials: dict name -> (r,g,b,a)."""
    used = sorted({m for o in objects for m in o.groups})
    out = []
    w = out.append
    w('<?xml version="1.0" encoding="utf-8"?>')
    w('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">')
    w('<asset><contributor><authoring_tool>bobs_truck meshlib</authoring_tool></contributor>'
      '<unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>')
    w('<library_effects>')
    for m in used:
        c = materials.get(m, (0.8, 0.8, 0.8, 1.0))
        w(f'<effect id="{m}-effect"><profile_COMMON><technique sid="common"><phong>'
          f'<diffuse><color sid="diffuse">{c[0]:.4f} {c[1]:.4f} {c[2]:.4f} {c[3]:.4f}</color></diffuse>'
          f'</phong></technique></profile_COMMON></effect>')
    w('</library_effects>')
    w('<library_materials>')
    for m in used:
        w(f'<material id="{m}-material" name="{m}"><instance_effect url="#{m}-effect"/></material>')
    w('</library_materials>')
    w('<library_geometries>')
    for o in objects:
        pos, nrm, uvs = [], [], []
        prims = []
        for m, tris in o.groups.items():
            idx = []
            for t in tris:
                for k in range(3):
                    vi = len(pos)
                    pos.append(t[k])
                    nrm.append(t[3 + k])
                    uvs.append(t[6 + k])
                    idx.append(vi)
            prims.append((m, len(tris), idx))
        gid = f"{o.name}-mesh"
        w(f'<geometry id="{gid}" name="{o.name}"><mesh>')
        w(f'<source id="{gid}-positions"><float_array id="{gid}-positions-array" count="{len(pos) * 3}">'
          + " ".join(f"{c:.5f}" for p in pos for c in p) + '</float_array>'
          f'<technique_common><accessor source="#{gid}-positions-array" count="{len(pos)}" stride="3">'
          '<param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>'
          '</accessor></technique_common></source>')
        w(f'<source id="{gid}-normals"><float_array id="{gid}-normals-array" count="{len(nrm) * 3}">'
          + " ".join(f"{c:.4f}" for p in nrm for c in p) + '</float_array>'
          f'<technique_common><accessor source="#{gid}-normals-array" count="{len(nrm)}" stride="3">'
          '<param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>'
          '</accessor></technique_common></source>')
        w(f'<source id="{gid}-map-0"><float_array id="{gid}-map-0-array" count="{len(uvs) * 2}">'
          + " ".join(f"{p[0]:.4f} {p[1]:.4f}" for p in uvs) + '</float_array>'
          f'<technique_common><accessor source="#{gid}-map-0-array" count="{len(uvs)}" stride="2">'
          '<param name="S" type="float"/><param name="T" type="float"/>'
          '</accessor></technique_common></source>')
        w(f'<vertices id="{gid}-vertices"><input semantic="POSITION" source="#{gid}-positions"/></vertices>')
        for m, cnt, idx in prims:
            w(f'<triangles material="{m}-material" count="{cnt}">'
              f'<input semantic="VERTEX" source="#{gid}-vertices" offset="0"/>'
              f'<input semantic="NORMAL" source="#{gid}-normals" offset="1"/>'
              f'<input semantic="TEXCOORD" source="#{gid}-map-0" offset="2" set="0"/>'
              '<p>' + " ".join(f"{i} {i} {i}" for i in idx) + '</p></triangles>')
        w('</mesh></geometry>')
    w('</library_geometries>')
    w('<library_visual_scenes><visual_scene id="Scene" name="Scene">')
    for o in objects:
        w(f'<node id="{o.name}" name="{o.name}" type="NODE">'
          '<matrix sid="transform">1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>'
          f'<instance_geometry url="#{o.name}-mesh" name="{o.name}"><bind_material><technique_common>')
        for m in o.groups:
            w(f'<instance_material symbol="{m}-material" target="#{m}-material">'
              '<bind_vertex_input semantic="UVMap" input_semantic="TEXCOORD" input_set="0"/></instance_material>')
        w('</technique_common></bind_material></instance_geometry></node>')
    w('</visual_scene></library_visual_scenes>')
    w('<scene><instance_visual_scene url="#Scene"/></scene>')
    w('</COLLADA>')
    with open(path, "w") as f:
        f.write("\n".join(out))


# ---------------------------------------------------------------- glTF binary writer
def write_glb(path, objects, materials, pbr=None):
    """Writes a GLB (Y-up per glTF spec: we rotate so +Z(up)->+Y and -Y(front)->+Z)."""
    pbr = pbr or {}
    used = sorted({m for o in objects for m in o.groups})
    mat_index = {m: i for i, m in enumerate(used)}
    gmats = []
    for m in used:
        c = materials.get(m, (0.8, 0.8, 0.8, 1.0))
        met, rough = pbr.get(m, (0.0, 0.6))
        gm = {"name": m, "pbrMetallicRoughness": {"baseColorFactor": list(c), "metallicFactor": met,
                                                   "roughnessFactor": rough}}
        if c[3] < 0.99:
            gm["alphaMode"] = "BLEND"
        gmats.append(gm)

    # BeamNG (x left, y back, z up) -> glTF Y-up: rotate -90deg about X: (x, y, z) -> (x, z, -y)
    def rot(p): return (p[0], p[2], -p[1])

    bin_parts = []
    accessors, views, meshes, nodes = [], [], [], []
    offset = 0

    def push(data_bytes, target):
        nonlocal offset
        pad = (-len(data_bytes)) % 4
        bin_parts.append(data_bytes + b"\x00" * pad)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data_bytes), "target": target})
        offset += len(data_bytes) + pad
        return len(views) - 1

    for o in objects:
        prims = []
        for m, tris in o.groups.items():
            pos, nrm = [], []
            for t in tris:
                for k in range(3):
                    pos.append(rot(t[k]))
                    nrm.append(rot(t[3 + k]))
            n = len(pos)
            pb = struct.pack(f"<{n * 3}f", *[c for p in pos for c in p])
            nb = struct.pack(f"<{n * 3}f", *[c for p in nrm for c in p])
            vp = push(pb, 34962)
            vn = push(nb, 34962)
            mn = [min(p[i] for p in pos) for i in range(3)]
            mx = [max(p[i] for p in pos) for i in range(3)]
            accessors.append({"bufferView": vp, "componentType": 5126, "count": n, "type": "VEC3", "min": mn, "max": mx})
            ap = len(accessors) - 1
            accessors.append({"bufferView": vn, "componentType": 5126, "count": n, "type": "VEC3"})
            an = len(accessors) - 1
            prims.append({"attributes": {"POSITION": ap, "NORMAL": an}, "material": mat_index[m]})
        meshes.append({"name": o.name, "primitives": prims})
        nodes.append({"name": o.name, "mesh": len(meshes) - 1})
    gltf = {
        "asset": {"version": "2.0", "generator": "bobs_truck meshlib"},
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes, "meshes": meshes, "materials": gmats,
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": offset}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    binb = b"".join(bin_parts)
    total = 12 + 8 + len(js) + 8 + len(binb)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binb), 0x004E4942) + binb)
