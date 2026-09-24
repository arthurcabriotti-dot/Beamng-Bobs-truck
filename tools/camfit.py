"""Fit a pinhole camera to a reference photo from 2D<->3D point pairs, then overlay the model.

usage: python3 tools/camfit.py            (fits every photo in PHOTOS, writes preview/overlay_*.jpg)
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))
import build_truck as bt  # noqa: E402

# Pixel positions were read off gridded copies of the photos.
# 3D points use model coordinates (+X left, +Y back, +Z up). Hub faces sit ~0.12 m outboard of the
# wheel centre plane; tyre contact = bottom of the outer sidewall.
FL_HUB = (bt.TRACK_F + 0.115, 0.0, bt.TIRE_R)
RL_HUB = (bt.TRACK_R + 0.13, bt.WB, bt.TIRE_R)
FR_HUB = (-bt.TRACK_F - 0.115, 0.0, bt.TIRE_R)
RR_HUB = (-bt.TRACK_R - 0.13, bt.WB, bt.TIRE_R)

PHOTOS = {
    "side_left": {
        "size": (1600, 1200),
        "guess": {"eye": (6.0, 1.6, 1.2), "target": (0.0, 1.7, 0.9), "fov": 70},
        "points": [
            (FL_HUB, (302, 852)),
            (RL_HUB, (1270, 765)),
            ((bt.TRACK_F + 0.10, 0.0, 0.01), (302, 975)),
            ((bt.TRACK_R + 0.10, bt.WB, 0.01), (1270, 867)),
            ((bt.TRACK_F + 0.10, 0.0, 2 * bt.TIRE_R - 0.01), (300, 716)),
            ((bt.TRACK_R + 0.10, bt.WB, 2 * bt.TIRE_R - 0.01), (1270, 662)),
        ],
    },
    "front_left": {
        "size": (1600, 1200),
        "distortion": "radial",
        "guess": {"eye": (3.2, -2.6, 1.3), "target": (-0.3, 1.0, 0.9), "fov": 75},
        "points": [
            (FL_HUB, (848, 903)),
            (RL_HUB, (1230, 690)),
            ((bt.TRACK_F + 0.10, 0.0, 0.01), (812, 1073)),
            ((bt.TRACK_R + 0.10, bt.WB, 0.01), (1212, 776)),
            # (tyre tops are hidden inside the arches in this close shot)
            ((0.72, -0.79, 0.954), (550, 668)),  # left headlight centre
            ((0.72, -0.78, 0.775), (553, 745)),  # left parking lamp centre
            ((0.80, 1.04, 1.85), (962, 345)),    # windshield top, left
            ((-0.80, 1.04, 1.85), (655, 366)),   # windshield top, right
            ((0.97, 0.76, 1.38), (950, 492)),    # windshield base, left
            ((1.0, -0.72, 0.64), (630, 845)),    # front bumper left end, top
        ],
    },
    "rear_left": {
        "size": (1600, 1200),
        "guess": {"eye": (3.0, 7.0, 0.4), "target": (0.0, 2.0, 1.0), "fov": 75},
        "points": [
            (RL_HUB, (635, 895)),
            (FL_HUB, (125, 850)),
            ((bt.TRACK_R + 0.10, bt.WB, 0.01), (650, 1015)),
            ((bt.TRACK_F + 0.10, 0.0, 0.01), (145, 928)),
            ((bt.TRACK_R + 0.10, bt.WB, 2 * bt.TIRE_R - 0.01), (640, 768)),
            ((bt.TRACK_F + 0.10, 0.0, 2 * bt.TIRE_R - 0.01), (130, 770)),
        ],
    },
    "rear_right": {
        "size": (1600, 1200),
        "guess": {"eye": (-2.0, 7.0, 1.3), "target": (0.0, 2.0, 0.9), "fov": 75},
        "points": [
            (RR_HUB, (805, 855)),
            (FR_HUB, (1043, 690)),
            ((-bt.TRACK_F - 0.10, 0.0, 0.01), (1040, 750)),
            ((-bt.TRACK_F - 0.10, 0.0, 2 * bt.TIRE_R - 0.01), (1045, 615)),
            ((-0.915, 4.53, 1.17), (582, 700)),   # right tail lamp centre
            ((0.0, 4.548, 1.128), (372, 690)),   # centre of the CHEVROLET lettering
        ],
    },
    "rear": {
        "size": (1600, 1200),
        "distortion": False,
        "guess": {"eye": (0.6, 6.5, 1.2), "target": (0.0, 3.0, 1.0), "fov": 75},
        "points": [
            ((0.904, 4.537, 1.165), (232, 700)),   # left tail lamp centre
            ((0.904, 4.537, 0.935), (245, 768)),   # left tail lamp bottom
            ((0.84, 4.54, 1.39), (272, 492)),      # tailgate top, left corner
            ((-0.018, 4.714, 0.678), (705, 765)),  # plate corners
            ((-0.327, 4.714, 0.678), (843, 758)),
            ((-0.018, 4.714, 0.525), (712, 862)),
            ((-0.327, 4.714, 0.525), (850, 855)),
            ((0.64, 1.93, 1.43), (262, 440)),      # rear window, bottom corners
            ((-0.64, 1.93, 1.43), (640, 440)),
        ],
    },
}


def rot_from_angles(yaw, pitch, roll):
    """Camera basis (right, up, forward) from yaw (about Z), pitch, roll."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    f = np.array([cy * cp, sy * cp, sp])
    r0 = np.cross(f, [0, 0, 1.0])
    r0 /= np.linalg.norm(r0)
    u0 = np.cross(r0, f)
    cr, sr = math.cos(roll), math.sin(roll)
    r = cr * r0 + sr * u0
    u = -sr * r0 + cr * u0
    return r, u, f


def params_full(params):
    """Older 8-value parameter sets have no principal-point offset."""
    p = list(params)
    return p + [0.0] * (10 - len(p))


def project(params, P, size, distort=True):
    ex, ey, ez, yaw, pitch, roll, fl, k1, cx, cy = params_full(params)
    r, u, f = rot_from_angles(yaw, pitch, roll)
    rel = np.asarray(P, float) - np.array([ex, ey, ez])
    x, y = rel @ r / (rel @ f), rel @ u / (rel @ f)
    d = 1 + k1 * (x * x + y * y) if distort else 1.0   # radial lens distortion
    W, H = size
    return np.stack([W / 2 + cx + fl * x * d, H / 2 + cy - fl * y * d], axis=1)


def initial(guess, size):
    e = np.array(guess["eye"], float)
    d = np.array(guess["target"], float) - e
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(d[2], math.hypot(d[0], d[1]))
    fl = 0.5 * size[1] / math.tan(math.radians(guess["fov"]) / 2)
    return np.array([*e, yaw, pitch, 0.0, fl, 0.0, 0.0, 0.0])


def fit(photo):
    P = np.array([p for p, _ in photo["points"]], float)
    Q = np.array([q for _, q in photo["points"]], float)
    x0 = initial(photo["guess"], photo["size"])
    k_ok = len(P) >= 8 and photo.get("distortion", True) is not False   # fit lens distortion with enough points
    c_ok = len(P) >= 9 and photo.get("distortion", True) is True        # ... and an off-centre (cropped) image
    lo = [-np.inf] * 7 + [-0.6 if k_ok else -1e-9] + [-250 if c_ok else -1e-9] * 2
    hi = [np.inf] * 7 + [0.6 if k_ok else 1e-9] + [250 if c_ok else 1e-9] * 2
    res = least_squares(lambda p: (project(p, P, photo["size"]) - Q).ravel(), x0, bounds=(lo, hi))
    err = np.abs(res.fun).reshape(-1, 2)
    return res.x, float(np.sqrt((err ** 2).sum(1)).mean())


def camera_from_params(params):
    ex, ey, ez, yaw, pitch, roll, fl, k1, cx, cy = params_full(params)
    r, u, f = rot_from_angles(yaw, pitch, roll)
    return np.array([ex, ey, ez]), r, u, f, fl, k1, cx, cy


if __name__ == "__main__":
    from render import render_cam
    out = {}
    objs = bt.build_all()
    for name, ph in PHOTOS.items():
        params, err = fit(ph)
        out[name] = [float(v) for v in params]
        print(f"{name}: eye=({params[0]:.2f},{params[1]:.2f},{params[2]:.2f}) focal={params[6]:.0f}px k1={params[7]:+.3f} c=({params[8]:+.0f},{params[9]:+.0f})  mean err {err:.1f}px")
        img = Image.open(os.path.join(bt.ROOT, "reference", f"{name}.jpg")).convert("RGB").resize(ph["size"])
        ren, mask = render_cam(objs, camera_from_params(params), ph["size"], skip={"bt_plowblade"})
        over = Image.composite(Image.blend(img, ren, 0.55), img, mask)
        over.save(os.path.join(bt.ROOT, "preview", f"overlay_{name}.jpg"), quality=88)
    with open(os.path.join(bt.ROOT, "tools", "cameras.json"), "w") as f:
        json.dump(out, f, indent=1)
