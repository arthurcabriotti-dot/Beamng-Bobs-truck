"""Blender stage: turns the procedural truck into a finished model.

  * imports every mesh object from build_truck.py
  * welds, quads-ifies, bevels hard edges (rounded panel edges like real sheet metal) and smooths
  * sets up physically based materials, sky, sun and ground
  * renders previews + side-by-side comparisons from the cameras solved from Bob's photos
  * exports the finished meshes back to the BeamNG .dae and a textured .glb

usage: python3 tools/blender_build.py [--no-render] [--samples N] [--scale S]
"""
import json
import math
import os
import sys

import bpy  # noqa: I001  (bpy must be imported before bmesh/mathutils)
import bmesh
from mathutils import Matrix, Vector

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import build_truck as bt  # noqa: E402
from meshlib import MeshObject, write_dae  # noqa: E402

ROOT = bt.ROOT
PREVIEW = os.path.join(ROOT, "preview")

# object -> (bevel width, segments). Anything not listed is only smoothed.
BEVEL = {
    "bt_body_front": (0.012, 3), "bt_body_cab": (0.010, 3), "bt_bed": (0.012, 3), "bt_tailgate": (0.008, 3),
    "bt_bumper_F": (0.006, 2), "bt_bumper_R": (0.006, 2), "bt_toolbox": (0.005, 2),
    "bt_plowmount": (0.006, 2), "bt_plowblade": (0.008, 2), "bt_frame_mesh": (0.004, 1),
    "bt_engine_mesh": (0.010, 2), "bt_axle_F": (0.004, 1), "bt_axle_R": (0.004, 1),
}
SMOOTH_ANGLE = math.radians(35)


# ------------------------------------------------------------------ materials
def img(name):
    path = os.path.join(bt.VEH, name)
    im = bpy.data.images.get(name) or bpy.data.images.load(path)
    return im


def principled(name, color=(0.8, 0.8, 0.8, 1), metallic=0.0, rough=0.5, **kw):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = rough
    for k, v in kw.items():
        if k in bsdf.inputs:
            bsdf.inputs[k].default_value = v
    return m, nt, bsdf


def tex_node(nt, image, bsdf=None, to="Base Color", scale=None, noncolor=False):
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = image
    if noncolor:
        image.colorspace_settings.name = "Non-Color"
    if scale is not None:
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (scale, scale, 1)
        nt.links.new(coord.outputs["UV"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
    if bsdf is not None and to:
        nt.links.new(t.outputs["Color"], bsdf.inputs[to])
    return t


def bump_from(nt, bsdf, tex, strength=0.3):
    b = nt.nodes.new("ShaderNodeBump")
    b.inputs["Strength"].default_value = strength
    nt.links.new(tex.outputs["Color"], b.inputs["Height"])
    nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])


def make_materials():
    M = {}
    # faded single-stage red: slightly patchy roughness/colour so it doesn't look like plastic
    m, nt, b = principled("bt_paint", (0.42, 0.008, 0.010, 1), 0.0, 0.32)
    for key, val in (("Coat Weight", 0.25), ("Coat Roughness", 0.25)):
        if key in b.inputs:
            b.inputs[key].default_value = val
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["To Min"].default_value = 0.22
    ramp.inputs["To Max"].default_value = 0.46
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
    nt.links.new(ramp.outputs["Result"], b.inputs["Roughness"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (0.44, 0.008, 0.010, 1)
    mix.inputs["B"].default_value = (0.36, 0.014, 0.016, 1)
    nt.links.new(noise.outputs["Fac"], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    M["bt_paint"] = m

    M["bt_chrome"] = principled("bt_chrome", (0.95, 0.95, 0.96, 1), 1.0, 0.04)[0]
    M["bt_argent"] = principled("bt_argent", (0.55, 0.56, 0.57, 1), 0.8, 0.35)[0]
    M["bt_black"] = principled("bt_black", (0.012, 0.012, 0.012, 1), 0.0, 0.5)[0]
    M["bt_rubber"] = principled("bt_rubber", (0.02, 0.02, 0.02, 1), 0.0, 0.85)[0]
    M["bt_tire"] = principled("bt_tire", (0.022, 0.022, 0.022, 1), 0.0, 0.88)[0]
    m, nt, b = principled("bt_glass", (0.55, 0.62, 0.60, 1), 0.0, 0.0)
    b.inputs["Transmission Weight"].default_value = 1.0
    b.inputs["IOR"].default_value = 1.5
    M["bt_glass"] = m
    m, nt, b = principled("bt_headlight", (0.9, 0.9, 0.88, 1), 0.0, 0.02)
    b.inputs["Transmission Weight"].default_value = 0.85
    M["bt_headlight"] = m
    M["bt_lamp_clear"] = principled("bt_lamp_clear", (0.75, 0.74, 0.68, 1), 0.3, 0.15)[0]
    M["bt_taillight"] = principled("bt_taillight", (0.35, 0.01, 0.012, 1), 0.1, 0.12)[0]
    M["bt_amber"] = principled("bt_amber", (0.85, 0.35, 0.02, 1), 0.1, 0.15)[0]
    M["bt_steelwheel"] = principled("bt_steelwheel", (0.015, 0.015, 0.016, 1), 0.3, 0.28)[0]
    m, nt, b = principled("bt_rust", (0.28, 0.11, 0.045, 1), 0.2, 0.9)
    n2 = nt.nodes.new("ShaderNodeTexNoise")
    n2.inputs["Scale"].default_value = 40
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    mx.inputs["A"].default_value = (0.30, 0.12, 0.05, 1)
    mx.inputs["B"].default_value = (0.12, 0.05, 0.03, 1)
    nt.links.new(n2.outputs["Fac"], mx.inputs["Factor"])
    nt.links.new(mx.outputs["Result"], b.inputs["Base Color"])
    M["bt_rust"] = m
    for name, s, rough, metal, bump in (("bt_tread", None, 0.9, 0.0, 0.6), ("bt_tire_side", None, 0.85, 0.0, 0.25)):
        m, nt, b = principled(name, (0.03, 0.03, 0.03, 1), metal, rough)
        t = tex_node(nt, img(name + ".png"), b)
        bump_from(nt, b, t, bump)
        M[name] = m
    m, nt, b = principled("bt_diamond", (0.7, 0.7, 0.72, 1), 0.9, 0.32)
    t = tex_node(nt, img("bt_diamond.png"), b)
    bump_from(nt, b, t, 0.5)
    M["bt_diamond"] = m
    M["bt_frame"] = principled("bt_frame", (0.02, 0.02, 0.02, 1), 0.2, 0.55)[0]
    M["bt_under"] = principled("bt_under", (0.012, 0.011, 0.010, 1), 0.2, 0.8)[0]
    M["bt_engine"] = principled("bt_engine", (0.45, 0.16, 0.04, 1), 0.2, 0.5)[0]
    M["bt_interior"] = principled("bt_interior", (0.05, 0.045, 0.045, 1), 0.0, 0.75)[0]
    M["bt_seat"] = principled("bt_seat", (0.03, 0.028, 0.026, 1), 0.0, 0.55)[0]
    M["bt_plowred"] = principled("bt_plowred", (0.55, 0.02, 0.015, 1), 0.1, 0.3)[0]
    M["bt_yellow"] = principled("bt_yellow", (0.75, 0.5, 0.03, 1), 0.1, 0.4)[0]
    M["bt_uws"] = principled("bt_uws", (0.03, 0.18, 0.65, 1), 0.2, 0.35)[0]
    for name in ("bt_plate", "bt_gauges"):
        m, nt, b = principled(name, (1, 1, 1, 1), 0.2, 0.4)
        tex_node(nt, img(name + ".png"), b)
        M[name] = m
    m, nt, b = principled("bt_tailgate_text", (1, 1, 1, 1), 0.0, 0.5)
    t = tex_node(nt, img("bt_tailgate_text.png"), b)
    nt.links.new(t.outputs["Alpha"], b.inputs["Alpha"])
    M["bt_tailgate_text"] = m
    return M


# ------------------------------------------------------------------ meshes
def to_blender(mo, mats):
    names = list(mo.groups)
    verts, faces, uvs, midx = [], [], [], []
    for mi, m in enumerate(names):
        for t in mo.groups[m]:
            base = len(verts)
            verts += [t[0], t[1], t[2]]
            faces.append((base, base + 1, base + 2))
            uvs.append((t[6], t[7], t[8]))
            midx.append(mi)
    me = bpy.data.meshes.new(mo.name)
    me.from_pydata(verts, [], faces)
    for m in names:
        me.materials.append(mats[m])
    uvl = me.uv_layers.new(name="UVMap")
    for p, (fuv, mi) in enumerate(zip(uvs, midx)):
        poly = me.polygons[p]
        poly.material_index = mi
        for k, li in enumerate(poly.loop_indices):
            uvl.data[li].uv = fuv[k]
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0004)
    bmesh.ops.join_triangles(bm, faces=bm.faces, angle_face_threshold=math.radians(2),
                             angle_shape_threshold=math.radians(40), cmp_materials=True, cmp_uvs=False)
    bm.to_mesh(me)
    bm.free()
    me.validate()
    for p in me.polygons:
        p.use_smooth = True
    me.set_sharp_from_angle(angle=SMOOTH_ANGLE)
    ob = bpy.data.objects.new(mo.name, me)
    bpy.context.scene.collection.objects.link(ob)
    if mo.name in BEVEL:
        w, seg = BEVEL[mo.name]
        bev = ob.modifiers.new("bevel", "BEVEL")
        bev.width = w
        bev.segments = seg
        bev.limit_method = "ANGLE"
        bev.angle_limit = math.radians(40)
        bev.harden_normals = True
        bev.use_clamp_overlap = True
    return ob


def export_game_meshes(objs):
    """Evaluated (bevelled) Blender meshes -> triangle soup -> BeamNG .dae."""
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for ob in objs:
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        cn = me.corner_normals
        uvl = me.uv_layers.active.data
        mo = MeshObject(ob.name)
        for lt in me.loop_triangles:
            mat = me.materials[lt.material_index].name
            p = [tuple(me.vertices[v].co) for v in lt.vertices]
            n = [tuple(cn[l].vector) for l in lt.loops]
            uv = [tuple(uvl[l].uv) for l in lt.loops]
            mo.groups.setdefault(mat, []).append((p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1], uv[2]))
        ev.to_mesh_clear()
        out.append(mo)
    cols = bt.preview_colors()[0]
    write_dae(os.path.join(bt.VEH, "bobs_truck.dae"), out, cols)
    return sum(o.tri_count() for o in out)


# ------------------------------------------------------------------ scene
def setup_world(sun_az=210, sun_el=35):
    sc = bpy.context.scene
    world = bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    sky = nt.nodes.new("ShaderNodeTexSky")
    for t in ("MULTIPLE_SCATTERING", "NISHITA", "HOSEK_WILKIE"):
        try:
            sky.sky_type = t
            break
        except TypeError:
            continue
    try:
        sky.sun_elevation = math.radians(sun_el)
        sky.sun_rotation = math.radians(sun_az)
    except AttributeError:
        pass
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.10
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 2.2
    sun.data.angle = math.radians(2)
    sun.rotation_euler = (math.radians(90 - sun_el), 0, math.radians(sun_az + 90))
    sc.collection.objects.link(sun)
    # driveway
    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 1.7, 0))
    ground = bpy.context.active_object
    m, nt2, b = principled("ground", (0.06, 0.06, 0.065, 1), 0.0, 0.8)
    n = nt2.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = 200
    bp = nt2.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = 0.15
    nt2.links.new(n.outputs["Fac"], bp.inputs["Height"])
    nt2.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    ground.data.materials.append(m)
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = -0.3


def add_camera(name, eye, r, u, f, focal_px, width_px, cx=0.0, cy=0.0):
    cam = bpy.data.cameras.new(name)
    cam.shift_x = -cx / width_px
    cam.shift_y = cy / width_px
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = 36.0
    cam.lens = focal_px * 36.0 / width_px
    cam.clip_start = 0.05
    ob = bpy.data.objects.new(name, cam)
    rot = Matrix((r, u, [-c for c in f])).transposed()
    ob.matrix_world = Matrix.Translation(Vector(eye)) @ rot.to_4x4()
    bpy.context.scene.collection.objects.link(ob)
    return ob


def look_cam(name, eye, target, fov=38, W=1600):
    import numpy as np
    e, t = np.array(eye, float), np.array(target, float)
    f = (t - e) / np.linalg.norm(t - e)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    H = W * 0.75
    focal = 0.5 * H / math.tan(math.radians(fov) / 2)
    return add_camera(name, eye, r, u, f, focal, W)


def render(cam, path, W, H, samples):
    sc = bpy.context.scene
    sc.camera = cam
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def render_thumbnails(samples):
    """Vehicle-selector thumbnails for each config (plus default.jpg)."""
    from PIL import Image
    configs = {
        "bobs_daily": {"bt_plowblade"},
        "bobs_plow": set(),
        "bobs_clean": {"bt_plowblade", "bt_plowmount", "bt_toolbox"},
    }
    cam = look_cam("thumb", (4.8, -5.4, 1.7), (0.0, 1.3, 0.9), 34, 1000)
    for cfg, hidden in configs.items():
        for ob in bpy.data.objects:
            if ob.name.startswith("bt_"):
                ob.hide_render = ob.name in hidden
        tmp = os.path.join(PREVIEW, f"_thumb_{cfg}.png")
        render(cam, tmp, 1000, 750, samples)
        im = Image.open(tmp).convert("RGB")
        im = im.crop((0, 94, 1000, 656)).resize((500, 281), Image.LANCZOS)
        im.save(os.path.join(bt.VEH, f"{cfg}.jpg"), quality=90)
        if cfg == "bobs_daily":
            im.save(os.path.join(bt.VEH, "default.jpg"), quality=90)
        os.remove(tmp)
    for ob in bpy.data.objects:
        ob.hide_render = False


def main():
    args = sys.argv[1:]
    do_render = "--no-render" not in args
    samples = int(args[args.index("--samples") + 1]) if "--samples" in args else 48
    scale = float(args[args.index("--scale") + 1]) if "--scale" in args else 0.5
    only = args[args.index("--only") + 1].split(",") if "--only" in args else None
    thumbs = "--thumbs" in args
    photo = "--photo" in args

    bpy.ops.wm.read_factory_settings(use_empty=True)
    objs_src = bt.build_all()
    bt.build_textures()
    bt.write_materials()
    mats = make_materials()
    objs = [to_blender(objs_src[k], mats) for k in sorted(objs_src)]
    if photo:
        import photo_texture
        by_name = {o.name: o for o in objs}
        res = photo_texture.run(by_name)
        photo_texture.assign_photo_materials(by_name, res)
    ntri = export_game_meshes(objs)
    print(f"exported {ntri} triangles to bobs_truck.dae")
    os.makedirs(PREVIEW, exist_ok=True)

    # plow blade only shows in its own render
    blade = bpy.data.objects["bt_plowblade"]
    bpy.ops.export_scene.gltf(filepath=os.path.join(PREVIEW, "bobs_truck.glb"), export_format="GLB",
                              use_selection=False, export_apply=True, export_lights=False, export_cameras=False)
    setup_world()
    if thumbs:
        render_thumbnails(samples)
    if not do_render:
        return

    from PIL import Image
    W, H = int(1600 * scale), int(1200 * scale)
    blade.hide_render = True
    cams = json.load(open(os.path.join(TOOLS, "cameras.json")))
    sys.path.insert(0, TOOLS)
    from camfit import camera_from_params
    for name, params in cams.items():
        if only and name not in only:
            continue
        eye, r, u, f, fl, k1, cx, cy = camera_from_params(params)
        cam = add_camera("cam_" + name, eye, r, u, f, fl, 1600, cx, cy)
        out = os.path.join(PREVIEW, f"blender_{name}.png")
        render(cam, out, W, H, samples)
        ph = Image.open(os.path.join(ROOT, "reference", f"{name}.jpg")).convert("RGB").resize((W, H))
        rd = Image.open(out).convert("RGB")
        both = Image.new("RGB", (2 * W, H))
        both.paste(ph, (0, 0))
        both.paste(rd, (W, 0))
        both.save(os.path.join(PREVIEW, f"compare_{name}.jpg"), quality=88)
    hero = {
        "hero_front": ((4.6, -5.2, 1.6), (0.0, 1.2, 0.95), 36),
        "hero_rear": ((-4.2, 9.6, 1.9), (0.0, 2.3, 0.95), 36),
        "hero_side": ((8.5, 1.8, 1.2), (0.0, 1.8, 0.95), 38),
    }
    for name, (eye, tgt, fov) in hero.items():
        if only and name not in only:
            continue
        render(look_cam(name, eye, tgt, fov, W), os.path.join(PREVIEW, f"{name}.png"), W, H, samples)
    if not only or "hero_plow" in only:
        blade.hide_render = False
        render(look_cam("plow", (4.6, -5.6, 1.7), (0.0, 1.0, 0.9), 38, W),
               os.path.join(PREVIEW, "hero_plow.png"), W, H, samples)


if __name__ == "__main__":
    main()
