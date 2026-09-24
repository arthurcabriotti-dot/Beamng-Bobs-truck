"""Minimal numpy z-buffer renderer for previewing the generated model (no GPU needed).

usage: python3 tools/render.py [out_dir]
"""
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import build_truck as bt  # noqa: E402

SKIP_DEFAULT = {"bt_plowblade"}
SKIP_MATS = {"bt_tailgate_text"}


def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = map(np.array, (eye, target, up))
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    return eye, r, u, f


def render(objs, eye, target, W=1200, H=800, fov=38, ss=2, skip=SKIP_DEFAULT, cols=None, bg=True):
    eye, r, u, f = look_at(eye, target)
    fl = 0.5 * H / math.tan(math.radians(fov) / 2)
    return render_core(objs, eye, r, u, f, fl, W, H, ss, skip, cols, bg, ground=True)[0]


def render_cam(objs, cam, size, skip=SKIP_DEFAULT, ss=2):
    """Render with an explicit camera (eye, right, up, forward, focal_px); returns (image, model mask)."""
    eye, r, u, f, fl = cam[:5]
    k1 = cam[5] if len(cam) > 5 else 0.0
    cxy = (cam[6], cam[7]) if len(cam) > 7 else (0.0, 0.0)
    return render_core(objs, np.asarray(eye), np.asarray(r), np.asarray(u), np.asarray(f), fl,
                       size[0], size[1], ss, skip, None, True, ground=False, k1=k1, cxy=cxy)


def render_core(objs, eye, r, u, f, fl, W, H, ss, skip, cols, bg, ground, k1=0.0, cxy=(0.0, 0.0)):
    cols = cols or bt.preview_colors()[0]
    Wi, Hi = W * ss, H * ss
    fl = fl * ss
    zbuf = np.full((Hi, Wi), np.inf, dtype=np.float32)
    img = np.zeros((Hi, Wi, 3), dtype=np.float32)
    # background: sky gradient + ground
    if bg:
        yy = np.linspace(0, 1, Hi)[:, None]
        img[:] = (np.array([0.62, 0.74, 0.88]) * (1 - yy) + np.array([0.85, 0.88, 0.9]) * yy)[:, None, :]
    light = np.array([0.35, 0.55, 0.75])
    light = light / np.linalg.norm(light)
    light2 = np.array([-0.5, -0.3, 0.4])
    light2 = light2 / np.linalg.norm(light2)

    tris, tcol, tnorm = [], [], []
    for name, o in objs.items():
        if name in skip:
            continue
        for m, ts in o.groups.items():
            if m in SKIP_MATS:
                continue
            c = np.array(cols.get(m, (0.8, 0.8, 0.8, 1))[:3])
            for t in ts:
                tris.append((t[0], t[1], t[2]))
                tcol.append(c)
                tnorm.append((t[3], t[4], t[5]))
    # ground quad
    gt = []
    step = 1.0
    for gx in (np.arange(-8, 8, step) if ground else []):
        for gy in np.arange(-7, 11, step):
            gt += [((gx, gy, 0), (gx + step, gy, 0), (gx + step, gy + step, 0)),
                   ((gx, gy, 0), (gx + step, gy + step, 0), (gx, gy + step, 0))]
    P = np.array(tris + gt, dtype=np.float64)  # (N,3,3)
    C = np.array(tcol + [np.array([0.36, 0.38, 0.33])] * len(gt))
    Nn = np.array(tnorm + [((0, 0, 1),) * 3] * len(gt), dtype=np.float64)
    rel = P - eye
    cx = rel @ r
    cy = rel @ u
    cz = rel @ f
    # shading per vertex (lambert + spec-ish), averaged
    nv = Nn / (np.linalg.norm(Nn, axis=2, keepdims=True) + 1e-9)
    view = -rel / (np.linalg.norm(rel, axis=2, keepdims=True) + 1e-9)
    # two-sided: flip normals facing away
    facing = np.sum(nv * view, axis=2, keepdims=True)
    nv = np.where(facing < 0, -nv, nv)
    diff = np.clip(nv @ light, 0, 1) * 0.75 + np.clip(nv @ light2, 0, 1) * 0.2 + 0.28
    hvec = view + light
    hvec /= np.linalg.norm(hvec, axis=2, keepdims=True)
    spec = np.clip(np.sum(nv * hvec, axis=2), 0, 1) ** 40
    shade = diff[..., None] * C[:, None, :] + spec[..., None] * 0.35
    near = 0.05
    for i in range(P.shape[0]):
        z = cz[i]
        if np.all(z < near):
            continue
        if np.any(z < near):
            continue  # skip clipping complexity (only ground could cross)
        xn, yn = cx[i] / z, cy[i] / z
        dd = 1 + k1 * (xn * xn + yn * yn)
        sx = Wi / 2 + cxy[0] * ss + fl * xn * dd
        sy = Hi / 2 + cxy[1] * ss - fl * yn * dd
        x0, x1 = int(max(0, math.floor(sx.min()))), int(min(Wi - 1, math.ceil(sx.max())))
        y0, y1 = int(max(0, math.floor(sy.min()))), int(min(Hi - 1, math.ceil(sy.max())))
        if x0 > x1 or y0 > y1:
            continue
        xs = np.arange(x0, x1 + 1) + 0.5
        ys = np.arange(y0, y1 + 1) + 0.5
        X, Y = np.meshgrid(xs, ys)
        (ax, bx, cxx), (ay, by, cyy) = sx, sy
        den = (by - cyy) * (ax - cxx) + (cxx - bx) * (ay - cyy)
        if abs(den) < 1e-12:
            continue
        w0 = ((by - cyy) * (X - cxx) + (cxx - bx) * (Y - cyy)) / den
        w1 = ((cyy - ay) * (X - cxx) + (ax - cxx) * (Y - cyy)) / den
        w2 = 1 - w0 - w1
        mask = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not mask.any():
            continue
        iz = w0 / z[0] + w1 / z[1] + w2 / z[2]
        depth = 1.0 / iz
        zb = zbuf[y0:y1 + 1, x0:x1 + 1]
        upd = mask & (depth < zb)
        if not upd.any():
            continue
        zb[upd] = depth[upd]
        sh = shade[i]
        col = (w0[..., None] * sh[0] / z[0] + w1[..., None] * sh[1] / z[1] + w2[..., None] * sh[2] / z[2]) * depth[..., None]
        blk = img[y0:y1 + 1, x0:x1 + 1]
        blk[upd] = col[upd]
    out = np.clip(img, 0, 1) ** (1 / 1.6)
    im = Image.fromarray((out * 255).astype(np.uint8))
    mask = Image.fromarray((np.isfinite(zbuf) * 255).astype(np.uint8))
    return im.resize((W, H), Image.LANCZOS), mask.resize((W, H), Image.LANCZOS)


VIEWS = {
    "front_34": ((-4.2, -5.8, 2.0), (0.0, 1.4, 0.9)),
    "rear_34": ((3.8, 9.2, 2.1), (0.0, 2.2, 0.9)),
    "side": ((9.5, 1.7, 1.1), (0.0, 1.7, 0.95)),
    "front": ((0.0, -8.0, 1.2), (0.0, 0.0, 0.95)),
    "rear": ((0.0, 11.5, 1.3), (0.0, 2.0, 0.9)),
    "top_34": ((5.5, -3.0, 5.5), (0.0, 1.7, 0.8)),
}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(bt.ROOT, "preview")
    only = sys.argv[2:] or list(VIEWS)
    os.makedirs(out, exist_ok=True)
    objs = bt.build_all()
    for name in only:
        eye, tgt = VIEWS[name]
        im = render(objs, eye, tgt)
        im.save(os.path.join(out, f"{name}.png"))
        print("wrote", name)
