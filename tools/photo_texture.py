"""Projects Bob's photos onto the model and bakes them into per-object textures (runs inside Blender).

For every textured object:
  1. unwrap a dedicated UV map ("PhotoUV")
  2. bake per-texel world position, normal and the procedural base colour (fallback)
For every photo (camera solved by camfit.py):
  3. render a depth map from that camera (shadow-map style visibility test)
  4. in numpy: project each texel into the photo, keep it if it is visible (depth test), inside the frame,
     not under a mask (things in the photo that aren't part of the truck), and not on a silhouette edge
     (where small camera errors would pick up background); weight by how square-on and close the view is
Then blend all photos per texel, fall back to the base colour where no photo sees the surface,
bleed the islands and save PNGs that become the BeamNG materials.
"""
import json
import math
import os
import sys

import bpy  # noqa: I001
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import build_truck as bt  # noqa: E402
from camfit import PHOTOS, camera_from_params, project  # noqa: E402

# object -> texture size
TEXTURED = {
    "bt_body_front": 4096, "bt_body_cab": 4096, "bt_bed": 4096, "bt_tailgate": 2048,
    "bt_bumper_F": 1024, "bt_bumper_R": 2048, "bt_toolbox": 1024, "bt_plowmount": 1024,
    "bt_wheel_FL": 512, "bt_wheel_FR": 512, "bt_wheel_RL": 512, "bt_wheel_RR": 512,
    "bt_tire_FL": 1024, "bt_tire_FR": 1024, "bt_tire_RL": 1024, "bt_tire_RR": 1024,
}
KEEP_MATS = {"bt_glass"}          # faces keeping their own material
PHOTO_WEIGHT = {"side_left": 1.0, "rear_left": 1.0, "rear_right": 1.0, "front_left": 0.5, "rear": 0.15}

# Regions of each photo that are NOT the truck even though they sit in front of it (image px, 1600x1200)
MASKS = {
    "side_left": [[(0, 0), (48, 0), (48, 1200), (0, 1200)],                        # plow frame at the far left
                  [(1375, 560), (1600, 560), (1600, 820), (1375, 820)]],           # plow blade / frame at right
    "rear_left": [[(1235, 480), (1600, 480), (1600, 1200), (1235, 1200)],          # detached plow frame
                  [(930, 900), (1180, 900), (1180, 1060), (930, 1060)]],            # board on the ground
    "rear_right": [[(0, 480), (300, 480), (300, 700), (200, 700), (200, 1200), (0, 1200)],  # lamp + hitch post
                   [(1200, 330), (1600, 330), (1600, 1200), (1200, 1200)]],       # flatbed truck
    "front_left": [[(0, 180), (345, 180), (345, 830), (0, 830)],                   # plow lights/pump + other truck
                   [(1250, 560), (1400, 560), (1400, 700), (1250, 700)]],          # plow blade on the lawn
    "rear": [[(718, 405), (1600, 405), (1600, 1200), (718, 1200)],                 # plow frame in front
             [(0, 990), (1600, 990), (1600, 1200), (0, 1200)]],                    # ground / board
}


def srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def img_to_np(im):
    w, h = im.size
    a = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(a)
    return a.reshape(h, w, 4)[::-1]      # top row first


# ------------------------------------------------------------------ UVs + bakes
def visible_faces(ob, cams):
    """Faces that at least one photo sees square-on enough to be worth texture space."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(ob, dg)
    me = ob.data
    vis = np.zeros(len(me.polygons), bool)
    centers = np.array([tuple(p.center) for p in me.polygons])
    normals = np.array([tuple(p.normal) for p in me.polygons])
    for params in cams.values():
        eye = camera_from_params(params)[0]
        q = project(params, centers, (1600, 1200))
        rel = centers - eye
        dist = np.linalg.norm(rel, axis=1)
        facing = np.sum(normals * -rel, axis=1) / np.maximum(dist, 1e-6) > 0.15
        inframe = (q[:, 0] > 0) & (q[:, 0] < 1600) & (q[:, 1] > 0) & (q[:, 1] < 1200)
        for i in np.nonzero(facing & inframe & ~vis)[0]:
            d = rel[i] / dist[i]
            hit = tree.ray_cast(Vector(eye), Vector(d), dist[i] + 0.01)
            if hit[0] is None or hit[2] == i or hit[3] > dist[i] - 0.01:
                vis[i] = True
    return vis


def unwrap(ob, cams=None, hidden_scale=0.18, size=2048, margin_px=6):
    """Own planar-chart unwrap + shelf packing (Blender's smart_project misbehaves headless).
    Faces are grouped into connected charts sharing a dominant normal axis, projected flat,
    faces no photo sees are shrunk, and everything is packed at the largest scale that fits."""
    import bmesh
    me = ob.data
    uvl = me.uv_layers.get("PhotoUV") or me.uv_layers.new(name="PhotoUV")
    me.uv_layers.active = uvl
    me.uv_layers["UVMap"].active_render = True
    vis = visible_faces(ob, cams) if cams else np.ones(len(me.polygons), bool)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    nf = len(bm.faces)
    key = []
    for f in bm.faces:
        n = f.normal
        a = max(range(3), key=lambda k: abs(n[k]))
        key.append((a, n[a] > 0, bool(vis[f.index])))
    parent = list(range(nf))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in bm.edges:
        lf = e.link_faces
        if len(lf) == 2 and key[lf[0].index] == key[lf[1].index]:
            ra, rb = find(lf[0].index), find(lf[1].index)
            if ra != rb:
                parent[ra] = rb
    charts = {}
    for f in bm.faces:
        charts.setdefault(find(f.index), []).append(f)
    items = []   # (faces, 2d loop coords dict, w, h)
    for faces in charts.values():
        a, _, v = key[faces[0].index]
        u_ax, v_ax = [k for k in range(3) if k != a]
        sc = 1.0 if v else hidden_scale
        pts = {l.index: (l.vert.co[u_ax] * sc, l.vert.co[v_ax] * sc) for f in faces for l in f.loops}
        arr = np.array(list(pts.values()))
        mn = arr.min(0)
        pts = {k: (x - mn[0], y - mn[1]) for k, (x, y) in pts.items()}
        w, h = np.ptp(arr, 0)
        if h > w:   # lie long charts flat for better shelf packing
            pts = {k: (y, w - x) for k, (x, y) in pts.items()}
            w, h = h, w
        items.append((faces, pts, max(w, 1e-4), max(h, 1e-4)))
    items.sort(key=lambda it: -it[3])
    m = margin_px / size

    def pack(s):
        x = y = rowh = 0.0
        place = []
        for faces, pts, w, h in items:
            ww, hh = w * s + m, h * s + m
            if ww > 1:
                return None
            if x + ww > 1:
                x, y, rowh = 0.0, y + rowh, 0.0
            place.append((x + m / 2, y + m / 2))
            x += ww
            rowh = max(rowh, hh)
        return place if y + rowh <= 1 else None
    lo, hi = 1e-3, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if pack(mid):
            lo = mid
        else:
            hi = mid
    place = pack(lo)
    lay = bm.loops.layers.uv["PhotoUV"]
    for (faces, pts, w, h), (ox, oy) in zip(items, place):
        for f in faces:
            for l in f.loops:
                px, py = pts[l.index]
                l[lay].uv = (ox + px * lo, oy + py * lo)
    bm.to_mesh(me)
    bm.free()
    return float(vis.mean())


def apply_modifiers(ob):
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    for m in list(ob.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


def emit_material(kind):
    m = bpy.data.materials.new(f"_bake_{kind}")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    nt.links.new(geo.outputs["Position" if kind == "pos" else "Normal"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    tex = nt.nodes.new("ShaderNodeTexImage")
    nt.nodes.active = tex
    return m, tex


def bake_object(ob, size):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 1
    sc.render.bake.margin = 6
    sc.render.bake.use_clear = True
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    out = {}
    orig = [s.material for s in ob.material_slots]
    for kind in ("pos", "nrm"):
        im = bpy.data.images.new(f"{ob.name}_{kind}", size, size, alpha=True, float_buffer=True)
        im.colorspace_settings.name = "Non-Color"
        m, tex = emit_material(kind)
        tex.image = im
        for s in ob.material_slots:
            s.material = m
        bpy.ops.object.bake(type="EMIT")
        out[kind] = img_to_np(im)
    for s, m in zip(ob.material_slots, orig):
        s.material = m
    # base colour of the procedural materials (fallback where no photo sees the surface)
    im = bpy.data.images.new(f"{ob.name}_base", size, size, alpha=True, float_buffer=True)
    im.colorspace_settings.name = "Non-Color"
    added = []
    for m in {s.material for s in ob.material_slots}:
        t = m.node_tree.nodes.new("ShaderNodeTexImage")
        t.image = im
        m.node_tree.nodes.active = t
        added.append((m, t))
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})
    out["base"] = img_to_np(im)
    for m, t in added:
        m.node_tree.nodes.remove(t)
    return out


# ------------------------------------------------------------------ depth maps
def depth_maps(cams, W=1600, H=1200):
    from blender_build import add_camera
    sc = bpy.context.scene
    mat = bpy.data.materials.new("_depth")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    cd = nt.nodes.new("ShaderNodeCameraData")
    nt.links.new(cd.outputs["View Z Depth"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    hidden = {}
    for ob in bpy.data.objects:
        hidden[ob.name] = ob.hide_render
        if ob.name in ("bt_glass", "bt_plowblade") or not ob.name.startswith("bt_"):
            ob.hide_render = True
    vl = sc.view_layers[0]
    vl.material_override = mat
    world_strength = None
    if sc.world:
        bg = sc.world.node_tree.nodes.get("Background")
        world_strength = bg.inputs["Strength"].default_value
        bg.inputs["Strength"].default_value = 0.0
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 1
    sc.cycles.use_denoising = False
    sc.cycles.pixel_filter_type = "BOX"
    sc.cycles.filter_width = 0.01
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    depths = {}
    for name, params in cams.items():
        eye, r, u, f, fl, k1, cx, cy = camera_from_params(params)
        cam = add_camera("_d_" + name, eye, r, u, f, fl, W, cx, cy)
        sc.camera = cam
        path = f"/tmp/_depth_{name}.exr"
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(path)
        d = img_to_np(im)[..., 0].copy()
        d[d <= 1e-4] = np.inf
        depths[name] = d
    vl.material_override = None
    for ob in bpy.data.objects:
        if ob.name in hidden:
            ob.hide_render = hidden[ob.name]
    if world_strength is not None:
        sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = world_strength
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "8"
    sc.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
    sc.cycles.filter_width = 1.5
    return depths


def edge_mask(depth, grow=7):
    """True near depth discontinuities (silhouettes) - camera misfit would sample background there."""
    d = np.where(np.isfinite(depth), depth, 1e3)
    gx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
    gy = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
    e = ((gx > 0.08 * d) | (gy > 0.08 * d)).astype(np.uint8) * 255
    im = Image.fromarray(e).filter(ImageFilter.MaxFilter(2 * grow + 1))
    return np.asarray(im) > 0


def photo_mask(name, W=1600, H=1200):
    m = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(m)
    for poly in MASKS.get(name, []):
        d.polygon(poly, fill=0)
    m = m.filter(ImageFilter.GaussianBlur(4))
    return np.asarray(m, np.float32) / 255.0


def sample(img, px, py):
    """Bilinear sample of HxWxC array at float pixel coords (pixel centres at +0.5)."""
    H, W = img.shape[:2]
    x = np.clip(px - 0.5, 0, W - 1.001)
    y = np.clip(py - 0.5, 0, H - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    a = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    b = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def bleed(rgb, cov, iters=12):
    """Push colour outward from covered texels so filtering at UV seams doesn't show gaps."""
    rgb, cov = rgb.copy(), cov.copy()
    for _ in range(iters):
        acc = np.zeros_like(rgb)
        cnt = np.zeros(cov.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            acc += np.roll(np.roll(rgb * cov[..., None], dy, 0), dx, 1)
            cnt += np.roll(np.roll(cov, dy, 0), dx, 1)
        new = (cnt > 0) & (cov == 0)
        rgb[new] = acc[new] / cnt[new][:, None]
        cov = np.where(new, 1.0, cov).astype(np.float32)
    return rgb


def run(objects_by_name):
    cams = json.load(open(os.path.join(TOOLS, "cameras.json")))
    cams = {k: v for k, v in cams.items() if k in PHOTO_WEIGHT}
    photos, masks = {}, {}
    for name in cams:
        im = Image.open(os.path.join(bt.ROOT, "reference", f"{name}.jpg")).convert("RGB").resize((1600, 1200))
        photos[name] = srgb_to_lin(np.asarray(im, np.float32) / 255.0)
        masks[name] = photo_mask(name)
    # textured objects: freeze the bevelled mesh, unwrap (texture space goes to what the photos see)
    for name in TEXTURED:
        ob = objects_by_name[name]
        apply_modifiers(ob)
        frac = unwrap(ob, cams, size=TEXTURED[name])
        print(f"  {name:16s} {frac * 100:4.0f}% of faces seen by a photo")
    depths = depth_maps(cams)
    edges = {n: edge_mask(d) for n, d in depths.items()}
    results = {}
    for name, size in TEXTURED.items():
        ob = objects_by_name[name]
        b = bake_object(ob, size)
        P, N, base = b["pos"][..., :3], b["nrm"][..., :3], b["base"][..., :3]
        cov = b["pos"][..., 3] > 0.5
        acc = np.zeros(P.shape, np.float32)
        wsum = np.zeros(P.shape[:2], np.float32)
        best = np.zeros(P.shape[:2], np.float32)
        for cname, params in cams.items():
            eye, r, u, f, fl, k1, cx, cy = camera_from_params(params)
            flat = P[cov]
            rel = flat - eye
            zc = rel @ f
            q_d = project(params, flat, (1600, 1200), distort=True)      # where it is in the photo
            q_u = project(params, flat, (1600, 1200), distort=False)     # where it is in the depth map
            inframe = (q_d[:, 0] > 8) & (q_d[:, 0] < 1592) & (q_d[:, 1] > 8) & (q_d[:, 1] < 1192) & (zc > 0.1)
            iu = np.clip(q_u[:, 0].astype(int), 0, 1599)
            iv = np.clip(q_u[:, 1].astype(int), 0, 1199)
            dmap = depths[cname][iv, iu]
            visible = inframe & (np.abs(dmap - zc) < 0.02 + 0.012 * zc)
            view = -rel / np.linalg.norm(rel, axis=1, keepdims=True)
            cosv = np.clip(np.sum(N[cov] * view, axis=1), 0, 1)
            m = sample(masks[cname][..., None], q_d[:, 0], q_d[:, 1])[:, 0]
            e = np.where(edges[cname][iv, iu], 0.03, 1.0)
            res = (fl / np.maximum(zc, 0.1)) / 300.0
            usable = visible * m * e * np.clip((cosv - 0.10) / 0.15, 0, 1)      # may this photo be used here?
            w = usable * cosv ** 3 * np.minimum(res, 3.0) ** 1.5 * PHOTO_WEIGHT[cname]  # how much we prefer it
            col = sample(photos[cname], q_d[:, 0], q_d[:, 1])
            full_w = np.zeros(P.shape[:2], np.float32)
            full_w[cov] = w
            full_use = np.zeros(P.shape[:2], np.float32)
            full_use[cov] = usable
            full_c = np.zeros(P.shape, np.float32)
            full_c[cov] = col
            # sharpen: strongly prefer the best view per texel (avoids ghosting from small misalignments)
            w4 = full_w ** 4
            acc += full_c * w4[..., None]
            wsum += w4
            best = np.maximum(best, full_use)
        photo = np.where(wsum[..., None] > 0, acc / np.maximum(wsum, 1e-12)[..., None], 0)
        alpha = best
        # bring the flat procedural colour to the photos' exposure where both exist
        good = (alpha > 0.9) & cov
        if good.sum() > 200:
            gain = np.median(photo[good].mean(1)) / max(1e-4, np.median(base[good].mean(1)))
            gain = float(np.clip(gain, 0.5, 2.5))
        else:
            gain = 1.0
        final = alpha[..., None] * photo + (1 - alpha[..., None]) * base * gain
        final = bleed(final, cov.astype(np.float32))
        path = os.path.join(bt.VEH, f"bt_photo_{name[3:]}.png")
        Image.fromarray((lin_to_srgb(final) * 255).astype(np.uint8)).save(path)
        results[name] = (path, float(alpha[cov].mean()))
        print(f"  {name:16s} photo coverage {alpha[cov].mean() * 100:5.1f}%")
    return results


def assign_photo_materials(objects_by_name, results):
    """Swap every non-glass material on textured objects for its baked photo material."""
    for name, (path, _) in results.items():
        ob = objects_by_name[name]
        mname = f"bt_photo_{name[3:]}"
        m = bpy.data.materials.new(mname)
        m.use_nodes = True
        nt = m.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        bsdf.inputs["Roughness"].default_value = 0.45
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(path)
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = "PhotoUV"
        nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        for s in ob.material_slots:
            if s.material and s.material.name.split(".")[0] not in KEEP_MATS:
                s.material = m
    # BeamNG material entries
    mp = os.path.join(bt.VEH, "main.materials.json")
    mats = json.load(open(mp))
    for name in results:
        mname = f"bt_photo_{name[3:]}"
        mats[mname] = {
            "name": mname, "mapTo": mname, "class": "Material",
            "Stages": [{"baseColorMap": bt.VPATH + f"{mname}.png", "baseColorFactor": [1, 1, 1, 1],
                        "metallicFactor": 0.0, "roughnessFactor": 0.45}, {}, {}, {}],
            "materialTag0": "beamng", "materialTag1": "vehicle", "version": 1.5,
        }
    with open(mp, "w") as f:
        json.dump(mats, f, indent=2)
