"""Builds the 3D model of Bob's 1985 Chevrolet K20 (square body, reg cab, long bed, 4x4).

Outputs (into mod/vehicles/bobs_truck):
  bobs_truck.dae          - all meshes, one object per flexbody
  main.materials.json     - BeamNG materials
  *.png                   - generated textures
and preview/bobs_truck.glb (viewable in any glTF viewer).

Coordinates (same as the jbeam): +X left, +Y rearward, +Z up, metres.
Front axle at y=0, rear axle at y=3.34 (131.5" wheelbase), ground at z=0.
"""
import json
import math
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(__file__))
from meshlib import (MeshObject, add, sub, mul, norm, cross, dot, lerp, bilerp, mirror_x,
                     write_dae, write_glb)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VEH = os.path.join(ROOT, "mod", "vehicles", "bobs_truck")
VPATH = "vehicles/bobs_truck/"

# ------------------------------------------------------------------ key dimensions
WB = 3.34            # wheelbase
TIRE_R = 0.405       # ~32" tire
TRACK_F = 1.69 / 2   # half track (wheel centre x)
TRACK_R = 1.67 / 2
HALF_W = 0.99        # body half width
BELT = 1.38          # beltline
ROOF = 1.85
BED_TOP = 1.40
BED_Y0, BED_Y1 = 1.955, 4.42
CAB_Y0, CAB_Y1 = 0.72, 1.93
ZC = 1.20            # shoulder crease height (runs fender -> door -> bed)
INSET = 0.025        # how far the shoulder leans in by the top edge

# ------------------------------------------------------------------ materials
# name: (rgba, metallic, roughness, extra)
MATS = {
    "bt_paint":        ((0.50, 0.035, 0.04, 1), 0.15, 0.50, {"instance": True}),
    "bt_chrome":       ((0.92, 0.92, 0.94, 1), 1.0, 0.07, {}),
    "bt_argent":       ((0.62, 0.63, 0.64, 1), 0.7, 0.35, {}),
    "bt_black":        ((0.018, 0.018, 0.018, 1), 0.0, 0.55, {}),
    "bt_rubber":       ((0.025, 0.025, 0.025, 1), 0.0, 0.85, {}),
    "bt_glass":        ((0.12, 0.14, 0.15, 0.28), 0.0, 0.05, {"translucent": True, "doubleSided": True}),
    "bt_headlight":    ((0.85, 0.86, 0.84, 1), 0.6, 0.15, {}),
    "bt_lamp_clear":   ((0.80, 0.80, 0.76, 1), 0.3, 0.2, {}),
    "bt_taillight":    ((0.55, 0.02, 0.02, 1), 0.2, 0.15, {}),
    "bt_amber":        ((0.95, 0.45, 0.02, 1), 0.2, 0.2, {}),
    "bt_steelwheel":   ((0.02, 0.02, 0.022, 1), 0.3, 0.3, {}),
    "bt_rust":         ((0.36, 0.16, 0.07, 1), 0.3, 0.9, {}),
    "bt_tire":         ((0.03, 0.03, 0.03, 1), 0.0, 0.9, {}),
    "bt_tire_side":    ((1, 1, 1, 1), 0.0, 0.85, {"tex": "bt_tire_side.png"}),
    "bt_tread":        ((1, 1, 1, 1), 0.0, 0.9, {"tex": "bt_tread.png"}),
    "bt_diamond":      ((1, 1, 1, 1), 0.85, 0.35, {"tex": "bt_diamond.png"}),
    "bt_frame":        ((0.03, 0.03, 0.03, 1), 0.2, 0.6, {}),
    "bt_under":        ((0.06, 0.055, 0.05, 1), 0.2, 0.8, {}),
    "bt_engine":       ((0.55, 0.22, 0.05, 1), 0.2, 0.5, {}),
    "bt_interior":     ((0.07, 0.065, 0.065, 1), 0.0, 0.8, {"doubleSided": True}),
    "bt_seat":         ((0.035, 0.032, 0.03, 1), 0.0, 0.6, {}),
    "bt_plowred":      ((0.62, 0.03, 0.02, 1), 0.2, 0.3, {}),
    "bt_yellow":       ((0.85, 0.62, 0.05, 1), 0.1, 0.4, {}),
    "bt_tailgate_text": ((1, 1, 1, 1), 0.0, 0.5, {"tex": "bt_tailgate_text.png", "alpha": True}),
    "bt_plate":        ((1, 1, 1, 1), 0.2, 0.4, {"tex": "bt_plate.png"}),
    "bt_uws":          ((0.05, 0.25, 0.75, 1), 0.2, 0.4, {}),
    "bt_gauges":       ((1, 1, 1, 1), 0.0, 0.4, {"tex": "bt_gauges.png"}),
}

OBJS = {}


def obj(name):
    if name not in OBJS:
        OBJS[name] = MeshObject(name)
    return OBJS[name]


# ------------------------------------------------------------------ helpers
def both_sides(o, fn):
    """Call fn(sign) for left (+1) and right (-1)."""
    fn(1)
    fn(-1)


def sx(s, x):
    return s * x


def box_s(o, mat, s, x0, x1, y0, y1, z0, z1, skip=()):
    """Box on side s (x given for the left side, mirrored for right)."""
    a, b = sorted((s * x0, s * x1))
    o.box(mat, (a, y0, z0), (b, y1, z1), skip=skip)


def quad_x(o, mat, x, y0, y1, z0, z1, facing, uv=None):
    """Axis-aligned quad in a plane of constant x facing +X (facing=1) or -X (-1)."""
    a, b, c, d = (x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)
    o.oriented_quad(mat, a, b, c, d, outward=(facing, 0, 0), uv=uv)


def quad_y(o, mat, y, x0, x1, z0, z1, facing, uv=None):
    a, b, c, d = (x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)
    o.oriented_quad(mat, a, b, c, d, outward=(0, facing, 0), uv=uv)


def sweep(o, mat, path, profile, cap=True):
    """Sweep a closed 2D profile (d, z) along a plan-view path [(x, y), ...].
    d is the offset along the path's right-hand normal (tangent (tx,ty) -> normal (ty,-tx))."""
    rings = []
    n = len(path)
    for i in range(n):
        if i == 0:
            t = (path[1][0] - path[0][0], path[1][1] - path[0][1])
        elif i == n - 1:
            t = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
        else:
            t0 = norm((path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1], 0))
            t1 = norm((path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1], 0))
            t = (t0[0] + t1[0], t0[1] + t1[1])
        tl = math.hypot(*t)
        t = (t[0] / tl, t[1] / tl)
        nrm = (t[1], -t[0])
        ring = [(path[i][0] + nrm[0] * d, path[i][1] + nrm[1] * d, z) for d, z in profile]
        rings.append(ring)
    m = len(profile)
    cen_prof = (sum(p[0] for p in profile) / m, sum(p[1] for p in profile) / m)
    for i in range(n - 1):
        ra, rb = rings[i], rings[i + 1]
        # centre of the swept tube at this segment, to orient faces outward
        ca = add(mul(add(ra[0], rb[0]), 0.0), (0, 0, 0))
        for j in range(m):
            k = (j + 1) % m
            q = [ra[j], ra[k], rb[k], rb[j]]
            # outward: from local profile centre to edge midpoint
            pc = lerp(centre_of(ra), centre_of(rb), 0.5)
            mid = lerp(lerp(ra[j], ra[k], 0.5), lerp(rb[j], rb[k], 0.5), 0.5)
            o.oriented_quad(mat, *q, outward=sub(mid, pc))
    if cap:
        from meshlib import triangulate
        for ring, sgn in ((rings[0], -1), (rings[-1], 1)):
            tris = triangulate(profile)
            i = 0 if sgn < 0 else n - 1
            j = 1 if sgn < 0 else n - 2
            want = sub(centre_of(rings[i]), centre_of(rings[j]))
            for a, b, c in tris:
                t = [ring[a], ring[b], ring[c]]
                fn = cross(sub(t[1], t[0]), sub(t[2], t[0]))
                if dot(fn, want) < 0:
                    t = [t[0], t[2], t[1]]
                o.tri(mat, *t)


def centre_of(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts), sum(p[2] for p in pts) / len(pts))


def framed_face(o, q, outward, win, thick, outer_mat, inner_mat="bt_interior", jamb_mat="bt_black",
                glass=True, trim=None, dividers=()):
    """A panel with a window hole. q = 4 corners (a->b = u, a->d = v).
    win = (u0, u1, v0, v1) or None. Builds outer skin, inner skin, jambs and glass."""
    inward = mul(norm(outward), -thick)
    qi = [add(p, inward) for p in q]
    if win is None:
        o.oriented_quad(outer_mat, *q, outward=outward)
        o.oriented_quad(inner_mat, *qi, outward=mul(outward, -1))
        return
    u0, u1, v0, v1 = win
    uu = [0, u0, u1, 1]
    vv = [0, v0, v1, 1]
    for layer, mat, out in ((q, outer_mat, outward), (qi, inner_mat, mul(outward, -1))):
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                cell = [bilerp(layer, uu[i], vv[j]), bilerp(layer, uu[i + 1], vv[j]),
                        bilerp(layer, uu[i + 1], vv[j + 1]), bilerp(layer, uu[i], vv[j + 1])]
                o.oriented_quad(mat, *cell, outward=out)
    w_out = [bilerp(q, u0, v0), bilerp(q, u1, v0), bilerp(q, u1, v1), bilerp(q, u0, v1)]
    w_in = [add(p, inward) for p in w_out]
    wc = centre_of(w_out)
    for k in range(4):
        a, b = w_out[k], w_out[(k + 1) % 4]
        c, d = w_in[(k + 1) % 4], w_in[k]
        mid = lerp(a, b, 0.5)
        o.oriented_quad(jamb_mat, a, b, c, d, outward=sub(wc, mid))
    if glass:
        g = [add(p, mul(inward, 0.35)) for p in w_out]
        obj("bt_glass").oriented_quad("bt_glass", *g, outward=outward)
    if trim:
        # thin bright ring just proud of the skin around the window
        off = mul(norm(outward), 0.003)
        tw = 0.012
        for k in range(4):
            a, b = w_out[k], w_out[(k + 1) % 4]
            inset_dir = norm(sub(wc, lerp(a, b, 0.5)))
            a2, b2 = add(a, mul(inset_dir, -tw)), add(b, mul(inset_dir, -tw))
            o.oriented_quad(trim, add(a2, off), add(b2, off), add(b, off), add(a, off), outward=outward)
    for du in dividers:
        # vertical bar across the window at u=du
        uu0, uu1 = u0 + (u1 - u0) * du - 0.004, u0 + (u1 - u0) * du + 0.004
        bar = [bilerp(q, uu0, v0), bilerp(q, uu1, v0), bilerp(q, uu1, v1), bilerp(q, uu0, v1)]
        bar = [add(p, mul(inward, 0.25)) for p in bar]
        o.oriented_quad("bt_chrome", *bar, outward=outward)
        o.oriented_quad("bt_chrome", *bar, outward=mul(outward, -1))


def x_side(z, ztop, xo=None):
    """Outer body-side x at height z (vertical below the crease, leaning in above it)."""
    xo = HALF_W if xo is None else xo
    if z <= ZC:
        return xo
    return xo - INSET * min(1.0, (z - ZC) / max(1e-6, ztop - ZC))


def shoulder(o, s, y0, y1, zt0, zt1, xin, xo=None, mat="bt_paint", zc0=None, zc1=None):
    """Solid upper band of a body side between y0..y1: from the crease (zc0/zc1, default ZC) up to
    the top edge (zt0 at y0, zt1 at y1), leaning inward by INSET."""
    xo = HALF_W if xo is None else xo
    zc0 = ZC if zc0 is None else zc0
    zc1 = ZC if zc1 is None else zc1
    p = [[[None, None], [None, None]], [[None, None], [None, None]]]
    for j, (y, zt, zc) in enumerate(((y0, zt0, zc0), (y1, zt1, zc1))):
        p[0][j][0], p[0][j][1] = (s * xin, y, zc), (s * xin, y, zt)
        p[1][j][0], p[1][j][1] = (s * xo, y, zc), (s * (xo - INSET), y, zt)
    o.hexa(mat, p)


def seam(o, s, y, z0, z1, ztop, w=0.003, mat="bt_black"):
    """Thin dark panel-gap line on the body side, following the shoulder lean."""
    zs = [z0] + ([ZC] if z0 < ZC < z1 else []) + [z1]
    for za, zb in zip(zs, zs[1:]):
        xa, xb = s * (x_side(za, ztop) + 0.0015), s * (x_side(zb, ztop) + 0.0015)
        o.oriented_quad(mat, (xa, y - w, za), (xa, y + w, za), (xb, y + w, zb), (xb, y - w, zb), outward=(s, 0, 0))


def rounded_rect(cx, cy, w, h, r, n=4):
    pts = []
    for (qx, qy, a0) in ((cx + w / 2 - r, cy + h / 2 - r, 0), (cx - w / 2 + r, cy + h / 2 - r, 90),
                         (cx - w / 2 + r, cy - h / 2 + r, 180), (cx + w / 2 - r, cy - h / 2 + r, 270)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((qx + r * math.cos(a), qy + r * math.sin(a)))
    return pts


def arch_lip(o, s, pts, w, proud=0.005):
    """Flange around a wheel opening: a strip `w` wide on the body side, standing `proud` out."""
    yc = sum(p[0] for p in pts) / len(pts)
    zc = min(p[1] for p in pts)
    inner, outer = [], []
    for y, z in pts:
        d = (y - yc, z - zc)
        l = math.hypot(*d) or 1
        inner.append((y, z))
        outer.append((y + d[0] / l * w, z + d[1] / l * w))
    x0, x1 = s * HALF_W, s * (HALF_W + proud)
    for k in range(len(pts) - 1):
        a, b = inner[k], inner[k + 1]
        c, d = outer[k + 1], outer[k]
        o.oriented_quad("bt_paint", (x1, *a), (x1, *b), (x1, *c), (x1, *d), outward=(s, 0, 0))
        o.oriented_quad("bt_paint", (x0, *a), (x0, *b), (x1, *b), (x1, *a),
                        outward=(0, yc - (a[0] + b[0]) / 2, zc - (a[1] + b[1]) / 2))
        o.oriented_quad("bt_paint", (x0, *d), (x0, *c), (x1, *c), (x1, *d),
                        outward=(0, (c[0] + d[0]) / 2 - yc, (c[1] + d[1]) / 2 - zc))


def rust_patch(o, s, y0, y1, z0, z1, x=None):
    x = HALF_W + 0.0025 if x is None else x
    quad_x(o, "bt_rust", s * x, y0, y1, z0, z1, s)


# ------------------------------------------------------------------ front clip
def build_front():
    o = obj("bt_body_front")
    top = lambda y: 1.12 + (y + 0.80) * (1.33 - 1.12) / 1.52       # fender top line (front -> cowl)
    crease = lambda y: top(y) - 0.14
    fender = [(-0.74, 0.66), (-0.53, 0.66), (-0.53, 0.76), (-0.50, 0.86), (-0.43, 0.94), (-0.31, 0.99),
              (0.31, 0.99), (0.43, 0.94), (0.50, 0.86), (0.53, 0.76), (0.53, 0.57), (0.72, 0.57),
              (0.72, crease(0.72)), (-0.74, crease(-0.74))]
    in_arch = lambda a, b: "bt_under" if abs(a[0]) <= 0.531 and abs(b[0]) <= 0.531 else None
    for s in (1, -1):
        x0, x1 = sorted((s * 0.66, s * HALF_W))
        o.extrude_x("bt_paint", fender, x0, x1, side_mat=in_arch)
        quad_x(o, "bt_under", s * 0.665, -0.53, 0.53, 0.57, 0.99, s)            # wheel-well liner
        shoulder(o, s, -0.74, 0.72, top(-0.74), top(0.72), 0.66, zc0=crease(-0.74), zc1=crease(0.72))
        # front corner of the fender, outboard of the headlights
        o.extrude_x("bt_paint", [(-0.80, 0.66), (-0.74, 0.66), (-0.74, crease(-0.74)), (-0.80, crease(-0.80))],
                    *sorted((s * 0.865, s * HALF_W)))
        shoulder(o, s, -0.80, -0.74, top(-0.80), top(-0.74), 0.865, zc0=crease(-0.80), zc1=crease(-0.74))
        # arch lip: slightly proud flange following the wheel opening
        lip = [(-0.53, 0.66), (-0.53, 0.76), (-0.50, 0.86), (-0.43, 0.94), (-0.31, 0.99),
               (0.31, 0.99), (0.43, 0.94), (0.50, 0.86), (0.53, 0.76), (0.53, 0.57)]
        arch_lip(o, s, lip, 0.022)
        # amber side marker with chrome surround
        xm = s * (HALF_W + 0.002)
        quad_x(o, "bt_chrome", xm, -0.765, -0.585, 0.795, 0.860, s)
        quad_x(o, "bt_amber", s * (HALF_W + 0.004), -0.755, -0.595, 0.805, 0.850, s)
    # hood (covers the fender tops)
    hood = [(-0.815, top(-0.815) - 0.008), (0.72, top(0.72) - 0.005), (0.72, top(0.72) + 0.025),
            (-0.815, top(-0.815) + 0.022)]
    o.extrude_x("bt_paint", hood, -0.978, 0.978)
    # hood seam shadow lines along the fender tops
    for s in (1, -1):
        # a thin black strip slightly outboard of the hood edge
        o.oriented_quad("bt_black", (s * 0.979, -0.80, top(-0.80)), (s * 0.979, 0.72, top(0.72)),
                        (s * 0.979, 0.72, top(0.72) + 0.017), (s * 0.979, -0.80, top(-0.80) + 0.017), outward=(s, 0, 0))

    # ---- grille / headlight assembly (front face at y=-0.80); modelled at the old height and
    # squashed to the real 0.66 .. 1.195 m opening measured from the photos
    front_obj = o
    o = MeshObject("grille_tmp")
    yf = -0.815
    # outer frame bars (argent)
    o.box("bt_argent", (-0.865, yf, 1.195), (0.865, -0.76, 1.295))   # header
    o.box("bt_argent", (-0.865, yf, 0.700), (0.865, -0.76, 0.745))   # bottom
    for s in (1, -1):
        box_s(o, "bt_argent", s, 0.835, 0.865, yf, -0.76, 0.70, 1.295)   # outer post
        box_s(o, "bt_argent", s, 0.565, 0.605, yf, -0.76, 0.70, 1.195)   # grille/lamp divider
        box_s(o, "bt_argent", s, 0.605, 0.835, yf, -0.76, 0.975, 1.005)  # lamp divider
        # headlight bucket + sealed beam
        box_s(o, "bt_black", s, 0.605, 0.835, -0.77, -0.74, 1.005, 1.195)
        box_s(o, "bt_chrome", s, 0.612, 0.828, -0.785, -0.77, 1.012, 1.188)
        box_s(o, "bt_headlight", s, 0.620, 0.820, -0.792, -0.785, 1.020, 1.180)
        # parking / turn lamp
        box_s(o, "bt_black", s, 0.605, 0.835, -0.77, -0.74, 0.745, 0.975)
        box_s(o, "bt_lamp_clear", s, 0.620, 0.820, -0.785, -0.77, 0.765, 0.960)
    # grille insert: dark back plane + argent egg-crate
    o.box("bt_black", (-0.565, -0.745, 0.745), (0.565, -0.73, 1.195))
    for z in (0.890, 1.040):
        o.box("bt_argent", (-0.565, -0.805, z), (0.565, -0.745, z + 0.025))
    for x in (-0.285, 0.0, 0.285):
        o.box("bt_argent", (x - 0.012, -0.805, 0.745), (x + 0.012, -0.745, 1.195))
    # chevy bowtie in the grille centre
    bow = [(-0.07, 0.955), (-0.025, 0.955), (-0.02, 0.945), (0.02, 0.945), (0.025, 0.955), (0.07, 0.955),
           (0.06, 0.985), (0.025, 0.985), (0.02, 0.995), (-0.02, 0.995), (-0.025, 0.985), (-0.06, 0.985)]
    o.extrude_y("bt_argent", [(x, z) for x, z in bow], -0.82, -0.805)
    gz = lambda p: (p[0], p[1], 0.655 + (p[2] - 0.70) * (1.10 - 0.655) / (1.295 - 0.70))
    front_obj.merge(o.transformed(gz))
    o = front_obj
    # radiator core behind grille
    o.box("bt_under", (-0.62, -0.70, 0.72), (0.62, -0.64, 1.20))


# ------------------------------------------------------------------ cab
def build_cab():
    o = obj("bt_body_cab")
    t = 0.05
    x1, y0, y1, z0, z1 = HALF_W - INSET, CAB_Y0, CAB_Y1, 0.56, BELT
    # --- lower tub (outer skin); the outer 1.8cm of each side is the lower panel + shoulder
    o.box("bt_paint", (-x1, y0, z0), (x1, y1, z1), skip=("+z",))
    for s in (1, -1):
        box_s(o, "bt_paint", s, x1, HALF_W, y0, y1, z0, ZC)
        shoulder(o, s, y0, y1, z1, z1, x1)
    # inner skin (faces inward)
    xi, yi0, yi1, zi0 = x1 - t, y0 + t, y1 - t, z0 + 0.10
    inner = MeshObject("tmp")
    inner.box("bt_interior", (-xi, yi0, zi0), (xi, yi1, z1), skip=("+z",))
    for m, tris in inner.groups.items():  # flip to face inward
        for tr in tris:
            o.groups.setdefault(m, []).append((tr[0], tr[2], tr[1], mul(tr[3], -1), mul(tr[5], -1), mul(tr[4], -1),
                                               tr[6], tr[8], tr[7]))
    # top rim of the tub (beltline cap)
    for (a, b, c, d) in (
        ((-x1, y0, z1), (x1, y0, z1), (xi, yi0, z1), (-xi, yi0, z1)),
        ((x1, y0, z1), (x1, y1, z1), (xi, yi1, z1), (xi, yi0, z1)),
        ((x1, y1, z1), (-x1, y1, z1), (-xi, yi1, z1), (xi, yi1, z1)),
        ((-x1, y1, z1), (-x1, y0, z1), (-xi, yi0, z1), (-xi, yi1, z1)),
    ):
        o.oriented_quad("bt_paint", a, b, c, d, outward=(0, 0, 1))
    # cowl vent (black) between hood and windshield
    o.box("bt_black", (-0.93, 0.725, BELT - 0.002), (0.93, 0.795, BELT + 0.008))
    # parked wipers
    for (p0, p1) in (((0.02, 0.815, 1.395), (0.66, 0.835, 1.425)), ((-0.66, 0.815, 1.395), (-0.04, 0.835, 1.425))):
        o.cylinder("bt_black", p0, p1, 0.007, segs=6)
        o.cylinder("bt_black", add(p0, (0, 0.012, 0.02)), add(p1, (0, 0.012, 0.02)), 0.004, segs=6)

    # --- greenhouse
    fb, rb = 0.76, 1.93         # bottom front / rear y
    ft, rt = 1.04, 1.895        # top front / rear y
    xb, xt = 0.97, 0.80
    zb, zt = BELT, ROOF
    FL_b, FR_b = (xb, fb, zb), (-xb, fb, zb)
    RL_b, RR_b = (xb, rb, zb), (-xb, rb, zb)
    FL_t, FR_t = (xt, ft, zt), (-xt, ft, zt)
    RL_t, RR_t = (xt, rt, zt), (-xt, rt, zt)
    th = 0.035
    # windshield
    framed_face(o, [FR_b, FL_b, FL_t, FR_t], (0, -0.8, 0.6), (0.045, 0.955, 0.06, 0.93), th,
                "bt_paint", trim="bt_chrome")
    # rear window: 4 panes (sliding) with chrome frame
    framed_face(o, [RL_b, RR_b, RR_t, RL_t], (0, 1, 0.05), (0.17, 0.83, 0.10, 0.86), th,
                "bt_paint", trim="bt_chrome", dividers=(0.25, 0.5, 0.75))
    # sides with vent window divider
    for s in (1, -1):
        a, b, c, d = (s * xb, fb, zb), (s * xb, rb, zb), (s * xt, rt, zt), (s * xt, ft, zt)
        framed_face(o, [a, b, c, d], (s, 0, 0.3), (0.06, 0.86, 0.05, 0.90), th, "bt_paint",
                    trim="bt_chrome", dividers=(0.22,))
    # roof
    framed_face(o, [FL_t, RL_t, RR_t, FR_t], (0, 0, 1), None, th, "bt_paint")
    # drip rails (chrome) along roof sides
    for s in (1, -1):
        p0, p1 = (s * (xt + 0.012), ft + 0.01, zt - 0.012), (s * (xt + 0.012), rt, zt - 0.012)
        o.box("bt_chrome", (min(p0[0], s * (xt - 0.005)), ft - 0.02, zt - 0.02),
              (max(p0[0], s * (xt - 0.005)), rt, zt - 0.005))
    # roof marker? (none on this truck)

    # --- door details
    for s in (1, -1):
        xs = s * (HALF_W + 0.0015)
        for y in (0.745, 1.81):  # door seams
            seam(o, s, y, 0.58, BELT, BELT)
        quad_x(o, "bt_black", xs, 0.745, 1.81, 0.585, 0.591, s)
        # handle + lock (on the shoulder, so they sit a little further in)
        xh = x_side(1.30, BELT)
        box_s(o, "bt_chrome", s, xh - 0.004, xh + 0.016, 1.60, 1.73, 1.285, 1.315)
        box_s(o, "bt_black", s, xh - 0.004, xh + 0.010, 1.62, 1.71, 1.278, 1.286)
        box_s(o, "bt_chrome", s, xh - 0.004, xh + 0.006, 1.765, 1.785, 1.28, 1.30)
        # rocker shading + rust at the cab corner (as on the real truck)
        quad_x(o, "bt_under", xs, 0.72, CAB_Y1, 0.56, 0.58, s)
        rust_patch(o, s, 1.815, CAB_Y1 - 0.005, 0.565, 0.66)
        rust_patch(o, s, 1.75, 1.815, 0.565, 0.60)
        # mirror: two arms + head, chrome door bracket
        box_s(o, "bt_chrome", s, HALF_W - 0.01, 1.03, 0.875, 0.945, 1.29, 1.37)
        for z in (1.33, 1.43):
            box_s(o, "bt_chrome", s, HALF_W - 0.01, 1.14, 0.90, 0.915, z, z + 0.014)
        box_s(o, "bt_chrome", s, 1.10, 1.28, 0.88, 0.935, 1.36, 1.62)
        box_s(o, "bt_glass", s, 1.108, 1.272, 0.936, 0.938, 1.368, 1.612)
    # --- interior
    i = o
    i.box("bt_seat", (-0.86, 1.40, 0.92), (0.86, 1.83, 1.06))                    # cushion
    i.hexa("bt_seat", [[[(-0.86, 1.75, 1.06), (-0.86, 1.75 + 0.12, 1.66)],
                        [(-0.86, 1.89, 1.06), (-0.86, 1.89, 1.66)]],
                       [[(0.86, 1.75, 1.06), (0.86, 1.75 + 0.12, 1.66)],
                        [(0.86, 1.89, 1.06), (0.86, 1.89, 1.66)]]])                # backrest
    i.box("bt_interior", (-0.93, 0.78, 1.06), (0.93, 1.02, 1.36))                 # dash
    i.hexa("bt_interior", [[[(-0.93, 0.78, 1.36), (-0.93, 0.78, 1.40)], [(-0.93, 1.06, 1.33), (-0.93, 1.08, 1.37)]],
                           [[(0.93, 0.78, 1.36), (0.93, 0.78, 1.40)], [(0.93, 1.06, 1.33), (0.93, 1.08, 1.37)]]])
    quad_y(i, "bt_gauges", 1.0205, 0.20, 0.62, 1.16, 1.30, 1, uv=((1, 0), (0, 0), (0, 1), (1, 1)))
    # steering column + wheel (driver on the left, +X)
    col0, col1 = (0.40, 1.00, 1.14), (0.40, 1.24, 1.36)
    i.cylinder("bt_black", col0, col1, 0.035, segs=10)
    axis = norm(sub(col1, col0))
    i.torus("bt_black", add(col1, mul(axis, 0.02)), axis, 0.19, 0.016, segs=28, rsegs=8)
    for ang in (0, 2.1, 4.2):
        tmp = (0, 0, 1)
        u = norm(cross(axis, tmp)); v = cross(axis, u)
        tip = add(add(col1, mul(axis, 0.02)), add(mul(u, 0.18 * math.cos(ang)), mul(v, 0.18 * math.sin(ang))))
        i.cylinder("bt_black", add(col1, mul(axis, 0.02)), tip, 0.012, segs=6)
    # floor mat & shifter (column shift - skip), transmission hump
    i.box("bt_rubber", (-0.92, 0.80, 0.745), (0.92, 1.87, 0.75))
    i.box("bt_rubber", (-0.16, 0.80, 0.75), (0.16, 1.40, 0.86))


# ------------------------------------------------------------------ bed
def build_bed():
    o = obj("bt_bed")
    RC = 0.03              # rear corner radius (bed side wraps round into the tail lamp face)
    YS = BED_Y1 - RC       # bed side panel ends here, the rounded post takes over
    side = [(BED_Y0, 0.63), (2.80, 0.63), (2.80, 0.76), (2.83, 0.86), (2.90, 0.94), (3.02, 0.99),
            (3.66, 0.99), (3.78, 0.94), (3.85, 0.86), (3.88, 0.76), (3.88, 0.63), (YS, 0.63),
            (YS, ZC), (BED_Y0, ZC)]
    in_arch = lambda a, b: "bt_under" if 2.79 <= a[0] <= 3.89 and 2.79 <= b[0] <= 3.89 and max(a[1], b[1]) < 1.0 else None
    for s in (1, -1):
        x0, x1 = sorted((s * 0.91, s * HALF_W))
        o.extrude_x("bt_paint", side, x0, x1, side_mat=in_arch)
        shoulder(o, s, BED_Y0, YS, BED_TOP, BED_TOP, 0.91)
        arch_lip(o, s, [(2.80, 0.63), (2.80, 0.76), (2.83, 0.86), (2.90, 0.94), (3.02, 0.99),
                        (3.66, 0.99), (3.78, 0.94), (3.85, 0.86), (3.88, 0.76), (3.88, 0.63)], 0.022)
        quad_x(o, "bt_under", s * 0.905, 2.80, 3.88, 0.63, 0.99, s)             # wheel-well liner
        box_s(o, "bt_paint", s, 0.885, HALF_W - INSET + 0.003, BED_Y0, BED_Y1, BED_TOP, BED_TOP + 0.016)  # rail cap
        for yp in (2.16, 3.30, 4.24):                                            # stake pockets
            box_s(o, "bt_black", s, 0.905, 0.955, yp - 0.05, yp + 0.05, BED_TOP + 0.016, BED_TOP + 0.0175)
        # rounded rear post: plan view with a radiused outer/rear corner, lower + shoulder parts
        for (za, zb, xo) in ((0.63, ZC, HALF_W), (ZC, BED_TOP, HALF_W - INSET * 0.6)):
            plan = [(0.84, 4.28), (xo, 4.28), (xo, BED_Y1 - RC)]
            for k in range(1, 6):
                a = math.radians(90 * k / 6)
                plan.append((xo - RC + RC * math.cos(a), BED_Y1 - RC + RC * math.sin(a)))
            plan += [(xo - RC, BED_Y1), (0.84, BED_Y1)]
            if s < 0:
                plan = [(-x, y) for x, y in plan]
            o.extrude_z("bt_paint", plan, za, zb)
        # tail lamp: chrome bezel, red lens with ribs, clear backup lens
        box_s(o, "bt_chrome", s, 0.846, 0.962, BED_Y1, BED_Y1 + 0.012, 0.925, 1.325)
        box_s(o, "bt_taillight", s, 0.854, 0.954, BED_Y1 + 0.012, BED_Y1 + 0.017, 1.015, 1.315)
        for zr in range(6):
            z = 1.03 + zr * 0.048
            box_s(o, "bt_taillight", s, 0.856, 0.952, BED_Y1 + 0.017, BED_Y1 + 0.019, z, z + 0.012)
        box_s(o, "bt_lamp_clear", s, 0.854, 0.954, BED_Y1 + 0.012, BED_Y1 + 0.017, 0.935, 1.008)
        # rear side marker (red)
        quad_x(o, "bt_taillight", s * (HALF_W + 0.002), 4.20, 4.30, 0.88, 0.92, s)
        # fuel door (rounded, set in a dark gap)
        xs0, xs1 = sorted((s * HALF_W, s * (HALF_W + 0.003)))
        o.extrude_x("bt_black", rounded_rect(2.545, 0.855, 0.162, 0.192, 0.025), xs0, xs1)
        xs0, xs1 = sorted((s * (HALF_W + 0.003), s * (HALF_W + 0.005)))
        o.extrude_x("bt_paint", rounded_rect(2.545, 0.855, 0.150, 0.180, 0.02), xs0, xs1)
        # rust along the lower edges
        rust_patch(o, s, BED_Y0, BED_Y0 + 0.07, 0.635, 0.70)
        rust_patch(o, s, 4.30, BED_Y1 - RC, 0.635, 0.67)
    o.box("bt_paint", (-0.91, BED_Y0, 0.93), (0.91, BED_Y0 + 0.05, BED_TOP))     # front wall
    o.box("bt_paint", (-0.91, BED_Y0 + 0.05, 0.90), (0.91, 4.40, 0.95))           # floor
    for x in (-0.66, -0.33, 0.0, 0.33, 0.66):                                     # floor ribs
        o.box("bt_paint", (x - 0.025, BED_Y0 + 0.05, 0.95), (x + 0.025, 4.36, 0.962))
    # wheel tubs inside the bed
    for s in (1, -1):
        box_s(o, "bt_paint", s, 0.66, 0.91, 2.86, 3.82, 0.95, 1.12)
    # under-bed: dark liner above the wheels
    for s in (1, -1):
        box_s(o, "bt_under", s, 0.66, 0.985, 2.80, 3.88, 0.985, 0.995)

    # tailgate
    tg = obj("bt_tailgate")
    tg.box("bt_paint", (-0.84, 4.36, 0.66), (0.84, 4.44, 1.39))
    tg.box("bt_paint", (-0.82, 4.44, 0.72), (0.82, 4.447, 1.035))    # lower stamped panel
    tg.box("bt_paint", (-0.82, 4.44, 1.215), (0.82, 4.446, 1.34))    # upper stamped panel
    quad_y(tg, "bt_tailgate_text", 4.4475, -0.33, 0.33, 1.07, 1.185, 1,
           uv=((1, 0), (0, 0), (0, 1), (1, 1)))
    tg.box("bt_black", (-0.09, 4.44, 1.28), (0.09, 4.448, 1.325))      # handle
    tg.box("bt_chrome", (-0.84, 4.40, 1.385), (0.84, 4.445, 1.395))  # top cap


# ------------------------------------------------------------------ bumpers
def build_bumpers():
    f = obj("bt_bumper_F")
    path = [(1.00, -0.72), (0.985, -0.84), (0.93, -0.895), (-0.93, -0.895), (-0.985, -0.84), (-1.00, -0.72)]
    # tangent goes -x along the front, so the right normal (ty,-tx) points -y = forward. good.
    prof = [(-0.02, 0.645), (0.0, 0.645), (0.035, 0.65), (0.055, 0.625), (0.064, 0.535),
            (0.056, 0.43), (0.025, 0.405), (-0.02, 0.41)]
    path = [(x, y) for x, y in reversed(path)]  # left->right? ensure normal points forward
    sweep(f, "bt_chrome", _orient_path(path, (0, -1)), prof)
    # bumper brackets
    for s in (1, -1):
        box_s(f, "bt_frame", s, 0.38, 0.50, -0.88, -0.76, 0.48, 0.60)
    # front plate (hangs below the bumper, passenger side of centre)
    f.box("bt_black", (-0.42, -0.905, 0.28), (-0.08, -0.895, 0.455))
    quad_y(f, "bt_plate", -0.908, -0.405, -0.095, 0.29, 0.445, -1, uv=((0, 0), (1, 0), (1, 1), (0, 1)))

    r = obj("bt_bumper_R")
    path = [(1.00, 4.36), (0.985, 4.52), (0.93, 4.575), (-0.93, 4.575), (-0.985, 4.52), (-1.00, 4.36)]
    prof = [(-0.15, 0.692), (0.0, 0.692), (0.022, 0.676), (0.022, 0.640), (0.036, 0.630), (0.036, 0.585),
            (0.022, 0.575), (0.022, 0.520), (0.0, 0.498), (-0.15, 0.498)]
    sweep(r, "bt_chrome", _orient_path(path, (0, 1)), prof)
    # diamond plate step on top
    for (x0, x1) in ((-0.93, 0.93),):
        quad_top(r, "bt_diamond", x0, x1, 4.43, 4.57, 0.6935, tile=6)
    # licence plate (driver side of centre), recessed black mount
    # licence plate: just right (passenger side) of centre, as in the photos
    r.box("bt_black", (-0.345, 4.605, 0.515), (0.0, 4.612, 0.685))
    quad_y(r, "bt_plate", 4.6135, -0.327, -0.018, 0.525, 0.678, 1, uv=((1, 0), (0, 0), (0, 1), (1, 1)))
    # hitch receiver + cross tube
    r.box("bt_frame", (-0.40, 4.42, 0.44), (0.40, 4.49, 0.50))
    r.box("bt_frame", (-0.037, 4.40, 0.43), (0.037, 4.64, 0.505))
    r.box("bt_rubber", (-0.026, 4.641, 0.442), (0.026, 4.642, 0.493))
    for s in (1, -1):
        box_s(r, "bt_frame", s, 0.38, 0.50, 4.40, 4.50, 0.50, 0.62)


def _orient_path(path, want_normal):
    """Make sure the right-hand normal of the path points toward want_normal."""
    (x0, y0), (x1, y1) = path[len(path) // 2 - 1], path[len(path) // 2]
    t = (x1 - x0, y1 - y0)
    nrm = (t[1], -t[0])
    if nrm[0] * want_normal[0] + nrm[1] * want_normal[1] < 0:
        return list(reversed(path))
    return path


def quad_top(o, mat, x0, x1, y0, y1, z, tile=1.0):
    uv = ((x0 * tile, y0 * tile), (x1 * tile, y0 * tile), (x1 * tile, y1 * tile), (x0 * tile, y1 * tile))
    o.oriented_quad(mat, (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), outward=(0, 0, 1), uv=uv)


# ------------------------------------------------------------------ frame, axles, driveline
def build_chassis():
    fr = obj("bt_frame_mesh")
    for s in (1, -1):
        box_s(fr, "bt_frame", s, 0.40, 0.47, -0.86, 4.46, 0.48, 0.68)     # rails
        box_s(fr, "bt_under", s, 0.47, 0.75, 1.95, 2.70, 0.52, 0.74)      # fuel tanks (dual)
    for y in (-0.80, 0.95, 2.25, 4.40):
        fr.box("bt_frame", (-0.40, y - 0.04, 0.50), (0.40, y + 0.04, 0.60))
    # exhaust along the passenger rail
    fr.cylinder("bt_under", (-0.30, 0.50, 0.50), (-0.33, 2.20, 0.45), 0.030, segs=8)
    fr.box("bt_under", (-0.40, 2.20, 0.37), (-0.20, 2.90, 0.53))
    fr.cylinder("bt_under", (-0.30, 2.90, 0.45), (-0.60, 4.20, 0.48), 0.028, segs=8)
    # engine + transmission + transfer case
    en = obj("bt_engine_mesh")
    en.box("bt_engine", (-0.30, -0.55, 0.62), (0.30, 0.30, 1.05))
    en.hexa("bt_engine", [[[(-0.30, -0.50, 1.05), (-0.18, -0.48, 1.18)], [(-0.30, 0.28, 1.05), (-0.18, 0.26, 1.18)]],
                          [[(0.30, -0.50, 1.05), (0.18, -0.48, 1.18)], [(0.30, 0.28, 1.05), (0.18, 0.26, 1.18)]]])
    en.cylinder("bt_black", (0, -0.1, 1.18), (0, -0.1, 1.25), 0.20, segs=16)   # air cleaner
    en.box("bt_under", (-0.20, -0.45, 0.52), (0.20, 0.20, 0.62))                # oil pan
    en.box("bt_under", (-0.17, 0.30, 0.60), (0.17, 1.05, 0.90))                 # TH400
    en.box("bt_under", (-0.05, 1.00, 0.48), (0.32, 1.30, 0.72))                 # NP205 t-case
    en.box("bt_under", (-0.40, -0.64, 0.80), (0.40, -0.60, 1.15))              # fan shroud
    # front axle (Dana 60-ish): tube + pumpkin on the driver side
    fa = obj("bt_axle_F")
    fa.cylinder("bt_under", (-0.76, 0, TIRE_R), (0.76, 0, TIRE_R), 0.045, segs=12)
    fa.lathe("bt_under", (0.20, 0.0, TIRE_R), (0, 1, 0),
             [(0.0, -0.10), (0.12, -0.08), (0.14, 0.0), (0.12, 0.10), (0.0, 0.12)], segs=16)
    for s in (1, -1):  # knuckles
        box_s(fa, "bt_under", s, 0.68, 0.78, -0.07, 0.07, TIRE_R - 0.12, TIRE_R + 0.12)
    fa.cylinder("bt_under", (-0.755, 0.20, 0.42), (0.755, 0.20, 0.42), 0.018, segs=8)  # tie rod
    ra = obj("bt_axle_R")
    ra.cylinder("bt_under", (-0.76, WB, TIRE_R), (0.76, WB, TIRE_R), 0.05, segs=12)
    ra.lathe("bt_under", (0.0, WB, TIRE_R), (0, -1, 0),
             [(0.0, -0.12), (0.14, -0.10), (0.16, 0.0), (0.14, 0.10), (0.0, 0.12)], segs=16)
    # leaf springs (mapped to frame + axle so they flex)
    ls = obj("bt_springs")
    for s in (1, -1):
        for (ya, yb, yc) in ((-0.62, 0.0, 0.62), (WB - 0.72, WB, WB + 0.72)):
            p = [(ya, 0.54), (yb, TIRE_R + 0.06), (yc, 0.54)]
            for k in range(2):
                (y0, z0), (y1, z1) = p[k], p[k + 1]
                ls.hexa("bt_under", [[[(s * 0.43, y0, z0 - 0.03), (s * 0.43, y0, z0)], [(s * 0.43, y1, z1 - 0.03), (s * 0.43, y1, z1)]],
                                     [[(s * 0.50, y0, z0 - 0.03), (s * 0.50, y0, z0)], [(s * 0.50, y1, z1 - 0.03), (s * 0.50, y1, z1)]]])
        # shocks
        ls.cylinder("bt_black", (s * 0.52, 0.12, TIRE_R), (s * 0.52, 0.20, 0.80), 0.025, segs=8)
        ls.cylinder("bt_black", (s * 0.55, WB + 0.12, TIRE_R), (s * 0.55, WB - 0.05, 0.80), 0.025, segs=8)
    # driveshafts
    ls.cylinder("bt_under", (0.20, 1.20, 0.56), (0.0, WB - 0.12, TIRE_R + 0.02), 0.04, segs=10)
    ls.cylinder("bt_under", (0.26, 1.02, 0.56), (0.20, 0.10, TIRE_R + 0.02), 0.035, segs=10)


# ------------------------------------------------------------------ wheels
def make_wheel(rear):
    """Left-side wheel at origin, axis +X (outward)."""
    w = MeshObject("w")
    t = MeshObject("t")
    ax = (1, 0, 0)
    o = (0, 0, 0)
    side = [(0.205, -0.105), (0.25, -0.122), (0.31, -0.128), (0.36, -0.124), (0.388, -0.112), (0.402, -0.09)]
    t.lathe("bt_tire", o, ax, side, segs=48)
    t.lathe("bt_tread", o, ax, [(0.402, -0.09), (0.405, -0.065), (0.405, 0.065), (0.402, 0.09)],
            segs=48, uv_repeat=24)
    t.lathe("bt_tire_side", o, ax, [(p[0], -p[1]) for p in reversed(side)], segs=72)
    # rim
    w.lathe("bt_steelwheel", o, ax, [(0.216, 0.075), (0.219, 0.088), (0.210, 0.095), (0.199, 0.088),
                                     (0.197, 0.030), (0.180, 0.025), (0.150, 0.045), (0.078, 0.050),
                                     (0.072, 0.030)], segs=40)
    w.lathe("bt_steelwheel", o, ax, [(0.199, 0.02), (0.199, -0.085), (0.214, -0.094)], segs=40)
    w.lathe("bt_under", o, ax, [(0.175, -0.005), (0.0, -0.005)], segs=24)   # brake / backing plate
    # hub
    if rear:
        w.lathe("bt_rust", o, ax, [(0.105, 0.035), (0.105, 0.075), (0.06, 0.085), (0.052, 0.12),
                                   (0.035, 0.132), (0.0, 0.132)], segs=24)
    else:
        # locking hub: rusty body, chrome dial
        w.lathe("bt_rust", o, ax, [(0.072, 0.035), (0.070, 0.085), (0.062, 0.09)], segs=24)
        w.lathe("bt_chrome", o, ax, [(0.062, 0.09), (0.060, 0.108), (0.050, 0.114), (0.0, 0.116)], segs=24)
        w.box("bt_black", (0.114, -0.007, -0.030), (0.120, 0.007, 0.030))  # dial knob
    # 8 lug nuts
    for k in range(8):
        a = 2 * math.pi * k / 8
        c = (0.048, 0.0875 * math.cos(a), 0.0875 * math.sin(a))
        if rear:
            c = (0.078, 0.0875 * math.cos(a), 0.0875 * math.sin(a))
        w.cylinder("bt_rust", c, add(c, (0.02, 0, 0)), 0.0115, segs=6)
    # 8 round vent holes in the disc (dark insets)
    for k in range(8):
        a = 2 * math.pi * (k + 0.5) / 8
        c = (0.0405, 0.16 * math.cos(a), 0.16 * math.sin(a))
        w.lathe("bt_black", c, (1, 0.0, 0.0), [(0.018, 0.0), (0.0, 0.0)], segs=12)
    return w, t


def build_wheels():
    for name, (x, y), rear in (("FL", (TRACK_F, 0.0), False), ("FR", (-TRACK_F, 0.0), False),
                               ("RL", (TRACK_R, WB), True), ("RR", (-TRACK_R, WB), True)):
        w, t = make_wheel(rear)
        if x < 0:
            w, t = w.mirrored(), t.mirrored()
        mv = lambda p, x=x, y=y: (p[0] + x, p[1] + y, p[2] + TIRE_R)
        OBJS[f"bt_wheel_{name}"] = w.transformed(mv, name=f"bt_wheel_{name}")
        tt = t.transformed(mv, name=f"bt_tire_{name}")
        if x < 0:  # mirrored geometry would mirror the sidewall lettering; flip u back
            tt.groups["bt_tire_side"] = [tr[:6] + tuple((-u, v) for (u, v) in tr[6:9]) for tr in tt.groups["bt_tire_side"]]
        OBJS[f"bt_tire_{name}"] = tt


# ------------------------------------------------------------------ accessories
def build_toolbox():
    o = obj("bt_toolbox")
    y0, y1 = BED_Y0 + 0.03, BED_Y0 + 0.48
    zr = BED_TOP + 0.016
    h = 0.13
    o.box("bt_diamond", (-0.995, y0, zr), (0.995, y1, zr + h))
    o.box("bt_diamond", (-0.875, y0, 1.12), (0.875, y1 - 0.08, zr))
    o.box("bt_diamond", (-0.99, y0 - 0.005, zr + h), (0.99, y1 + 0.012, zr + h + 0.012))   # lid
    quad_y(o, "bt_black", y1 + 0.0125, -0.99, 0.99, zr + h - 0.004, zr + h, 1)            # lid seam
    for s in (1, -1):
        box_s(o, "bt_black", s, 0.50, 0.62, y1, y1 + 0.008, zr + 0.06, zr + 0.105)        # paddle latches
        box_s(o, "bt_chrome", s, 0.525, 0.595, y1 + 0.008, y1 + 0.014, zr + 0.07, zr + 0.095)
    o.box("bt_uws", (-0.14, y1, zr + 0.05), (-0.04, y1 + 0.004, zr + 0.095))          # UWS badge
    # fix diamond UVs to tile nicely (auto UV is metres; texture repeats every 0.25m)
    tris = o.groups["bt_diamond"]
    o.groups["bt_diamond"] = [tuple(list(tr[:6]) + [(u * 4, v * 4) for (u, v) in tr[6:9]]) for tr in tris]


def build_plowmount():
    """Western Unimount style push frame with light bar and plow lights (lamps sit at hood height)."""
    o = obj("bt_plowmount")
    for s in (1, -1):
        box_s(o, "bt_frame", s, 0.38, 0.46, -1.12, -0.86, 0.40, 0.48)       # push beams to frame horns
        box_s(o, "bt_frame", s, 0.335, 0.385, -1.155, -1.105, 0.40, 1.11)   # light towers
        for xs_ in (0.49, 0.57):   # two rusty threaded studs
            o.cylinder("bt_rust", (s * xs_, -1.12, 1.11), (s * xs_, -1.12, 1.17), 0.008, segs=6)
        box_s(o, "bt_black", s, 0.40, 0.66, -1.10, -0.96, 1.17, 1.40)        # housing
        box_s(o, "bt_amber", s, 0.41, 0.65, -1.107, -1.10, 1.345, 1.39)      # turn lens on top
        box_s(o, "bt_black", s, 0.41, 0.65, -1.107, -1.10, 1.335, 1.345)
        box_s(o, "bt_chrome", s, 0.41, 0.65, -1.106, -1.10, 1.18, 1.335)     # reflector
        box_s(o, "bt_headlight", s, 0.415, 0.645, -1.112, -1.106, 1.185, 1.33)
    o.box("bt_frame", (-0.62, -1.16, 1.08), (0.62, -1.10, 1.11))            # light bar
    o.box("bt_frame", (-0.52, -0.99, 0.36), (0.52, -0.87, 0.46))            # receiver (behind bumper)
    o.box("bt_yellow", (-0.30, -1.05, 0.62), (-0.06, -0.87, 0.84))          # hydraulic pump
    o.cylinder("bt_black", (-0.18, -1.05, 0.72), (-0.18, -1.12, 0.72), 0.04, segs=10)
    o.cylinder("bt_black", (-0.12, -1.02, 0.86), (-0.12, -1.02, 1.05), 0.025, segs=10)  # pump reservoir


def build_plow():
    """Red Western straight blade, ~2.3m wide, carried raised."""
    o = obj("bt_plowblade")
    # moldboard: curved (concave forward) profile swept across x
    prof = []
    for k in range(9):
        a = math.radians(-60 + 120 * k / 8)
        prof.append((-1.38 - 0.10 * math.cos(a) + 0.10, 0.52 + 0.40 * math.sin(a)))  # (y, z)
    front = prof
    back = [(y + 0.03, z) for (y, z) in reversed(prof)]
    poly = front + back
    o.extrude_x("bt_plowred", poly, -1.16, 1.16)
    # cutting edge
    o.box("bt_frame", (-1.16, -1.44, 0.14), (1.16, -1.40, 0.20))
    # ribs on the back
    for x in (-0.9, -0.45, 0.0, 0.45, 0.9):
        o.box("bt_plowred", (x - 0.02, -1.34, 0.20), (x + 0.02, -1.26, 0.86))
    # A-frame back to the mount
    for s in (1, -1):
        o.cylinder("bt_frame", (s * 0.55, -1.28, 0.45), (s * 0.30, -1.20, 0.40), 0.04, segs=8)
    o.box("bt_frame", (-0.35, -1.30, 0.40), (0.35, -1.26, 0.48))
    # yellow blade guides at the ends
    for s in (1, -1):
        o.cylinder("bt_yellow", (s * 1.10, -1.30, 0.90), (s * 1.10, -1.30, 1.45), 0.012, segs=8)


# ------------------------------------------------------------------ textures
def build_textures():
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    fontb = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    # diamond plate
    S = 256
    img = Image.new("RGB", (S, S), (170, 172, 175))
    d = ImageDraw.Draw(img)
    for gy in range(0, S + 64, 64):
        for gx in range(0, S + 64, 64):
            for (cx, cy, rot) in ((gx, gy, 1), (gx + 32, gy + 32, -1)):
                pts = [(cx + rot * 14, cy - 5), (cx + rot * 17, cy - 2), (cx - rot * 14, cy + 5), (cx - rot * 17, cy + 2)]
                d.polygon([(p[0] + 2, p[1] + 2) for p in pts], fill=(110, 112, 115))
                d.polygon(pts, fill=(222, 224, 226))
    img = img.filter(ImageFilter.SMOOTH)
    img.save(os.path.join(VEH, "bt_diamond.png"))

    # tire tread (u around circumference, v across)
    W, H = 128, 128
    img = Image.new("RGB", (W, H), (20, 20, 20))
    d = ImageDraw.Draw(img)
    for row in range(2):
        y0 = row * 64
        d.rectangle([4, y0 + 4, 58, y0 + 28], fill=(34, 34, 34))
        d.rectangle([70, y0 + 36, 124, y0 + 60], fill=(34, 34, 34))
        d.rectangle([4, y0 + 36, 40, y0 + 60], fill=(32, 32, 32))
        d.rectangle([88, y0 + 4, 124, y0 + 28], fill=(32, 32, 32))
    img = img.rotate(90)
    img.save(os.path.join(VEH, "bt_tread.png"))

    # tyre sidewall: raised lettering (Yokohama Geolandar H/T), v=0 at the tread shoulder
    W, H = 4096, 256
    img = Image.new("RGB", (W, H), (22, 22, 22))
    d = ImageDraw.Draw(img)
    fl = ImageFont.truetype(fontb, 92)
    fs = ImageFont.truetype(fontb, 40)
    for k, (txt, font, yy) in enumerate((("YOKOHAMA", fl, 40), ("GEOLANDAR  H/T", fl, 40),
                                         ("YOKOHAMA", fl, 40), ("GEOLANDAR  H/T", fl, 40))):
        tw = d.textlength(txt, font=font)
        cx = W * (k + 0.5) / 4
        d.text((cx - tw / 2 + 3, yy + 3), txt, font=font, fill=(12, 12, 12))
        d.text((cx - tw / 2, yy), txt, font=font, fill=(34, 34, 34))
        d.text((cx - 300, yy + 110), "LT245/75R16  120/116S  M+S", font=fs, fill=(28, 28, 28))
    d.rectangle([0, 2, W, 8], fill=(28, 28, 28))
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save(os.path.join(VEH, "bt_tire_side.png"))

    # tailgate lettering (embossed look, transparent background)
    W, H = 1024, 180
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    f = ImageFont.truetype(fontb, 140)
    d = ImageDraw.Draw(img)
    text = "CHEVROLET"
    tw = d.textlength(text, font=f)
    x = (W - tw) / 2
    d.text((x + 5, 22), text, font=f, fill=(0, 0, 0, 150))       # shadow
    d.text((x - 3, 12), text, font=f, fill=(255, 255, 255, 70))  # highlight
    d.text((x, 16), text, font=f, fill=(60, 0, 0, 110))          # face
    img.save(os.path.join(VEH, "bt_tailgate_text.png"))

    # Connecticut classic vehicle plate
    W, H = 512, 256
    img = Image.new("RGB", (W, H), (236, 240, 244))
    d = ImageDraw.Draw(img)
    for yy in range(H):
        c = int(236 - 30 * (yy / H))
        d.line([(0, yy), (W, yy)], fill=(c, c + 4, 250))
    d.rectangle([4, 4, W - 5, H - 5], outline=(20, 40, 110), width=6)
    f1 = ImageFont.truetype(fontb, 30)
    f2 = ImageFont.truetype(fontb, 120)
    f3 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 26)
    for txt, font, yy in (("CONNECTICUT", f1, 14), ("00-EXEK", f2, 58), ("Classic Vehicle", f3, 206)):
        tw = d.textlength(txt, font=font)
        d.text(((W - tw) / 2, yy), txt, font=font, fill=(20, 35, 100))
    img.save(os.path.join(VEH, "bt_plate.png"))

    # gauge cluster
    W, H = 512, 180
    img = Image.new("RGB", (W, H), (12, 12, 12))
    d = ImageDraw.Draw(img)
    for cx, r in ((130, 70), (300, 70), (440, 45)):
        d.ellipse([cx - r, 90 - r, cx + r, 90 + r], outline=(200, 200, 200), width=4, fill=(25, 25, 25))
        for k in range(9):
            a = math.radians(210 - k * 30)
            d.line([(cx + (r - 14) * math.cos(a), 90 - (r - 14) * math.sin(a)),
                    (cx + (r - 4) * math.cos(a), 90 - (r - 4) * math.sin(a))], fill=(230, 230, 230), width=3)
        d.line([(cx, 90), (cx + (r - 18) * math.cos(math.radians(200)), 90 - (r - 18) * math.sin(math.radians(200)))],
               fill=(240, 90, 20), width=4)
    img.save(os.path.join(VEH, "bt_gauges.png"))


def write_materials():
    out = {}
    for name, (c, met, rough, ex) in MATS.items():
        st = {"baseColorFactor": [round(v, 4) for v in c], "metallicFactor": met, "roughnessFactor": rough}
        if "tex" in ex:
            st["baseColorMap"] = VPATH + ex["tex"]
        if ex.get("instance"):
            st["instanceBaseColor"] = True
            st["clearCoatFactor"] = 1.0
            st["clearCoatRoughnessFactor"] = 0.12
            st["baseColorFactor"] = [1, 1, 1, 1]
        if ex.get("translucent"):
            st["opacityFactor"] = c[3]
        m = {
            "name": name, "mapTo": name, "class": "Material",
            "persistentId": str(uuid.uuid5(uuid.NAMESPACE_URL, "bobs_truck/" + name)),
            "Stages": [st, {}, {}, {}],
            "materialTag0": "beamng", "materialTag1": "vehicle",
            "version": 1.5,
        }
        if ex.get("translucent"):
            m["translucent"] = True
            m["translucentZWrite"] = False
        if ex.get("alpha"):
            m["alphaTest"] = True
            m["alphaRef"] = 40
            m["translucent"] = True
        if ex.get("doubleSided"):
            m["doubleSided"] = True
        if name in ("bt_chrome", "bt_paint", "bt_glass"):
            m["dynamicCubemap"] = True
        out[name] = m
    with open(os.path.join(VEH, "main.materials.json"), "w") as f:
        json.dump(out, f, indent=2)


REAR_STRETCH = 0.10   # photos show the bed/bumper ending ~10 cm further back than first modelled
STRETCH_Y0, STRETCH_Y1 = 3.95, 4.25


def stretch_rear(p):
    y = p[1]
    if y <= STRETCH_Y0:
        return p
    t = min(1.0, (y - STRETCH_Y0) / (STRETCH_Y1 - STRETCH_Y0))
    return (p[0], y + REAR_STRETCH * t, p[2])


def build_all():
    OBJS.clear()
    build_front()
    build_cab()
    build_bed()
    build_bumpers()
    build_chassis()
    build_wheels()
    build_toolbox()
    build_plowmount()
    build_plow()
    for name in ("bt_bed", "bt_tailgate", "bt_bumper_R", "bt_frame_mesh"):
        OBJS[name] = OBJS[name].transformed(stretch_rear)
    return OBJS


def preview_colors():
    cols, pbr = {}, {}
    for name, (c, met, rough, ex) in MATS.items():
        col = c
        if name == "bt_paint":
            col = (0.50, 0.035, 0.04, 1)
        if name == "bt_diamond":
            col = (0.68, 0.69, 0.70, 1)
        if name in ("bt_tread", "bt_tire_side"):
            col = (0.05, 0.05, 0.05, 1)
        if name == "bt_tailgate_text":
            col = (0.3, 0.01, 0.01, 1)
        if name == "bt_plate":
            col = (0.85, 0.88, 0.95, 1)
        if name == "bt_gauges":
            col = (0.05, 0.05, 0.05, 1)
        cols[name] = col
        pbr[name] = (met, rough)
    return cols, pbr


if __name__ == "__main__":
    os.makedirs(VEH, exist_ok=True)
    objs = build_all()
    cols, pbr = preview_colors()
    ordered = [objs[k] for k in sorted(objs)]
    write_dae(os.path.join(VEH, "bobs_truck.dae"), ordered, cols)
    build_textures()
    write_materials()
    os.makedirs(os.path.join(ROOT, "preview"), exist_ok=True)
    write_glb(os.path.join(ROOT, "preview", "bobs_truck.glb"), ordered, cols, pbr)
    total = sum(o.tri_count() for o in ordered)
    for o in ordered:
        print(f"  {o.name:18s} {o.tri_count():6d} tris")
    print(f"total {total} triangles, {len(ordered)} objects")
