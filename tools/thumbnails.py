"""Renders vehicle-selector thumbnails for each config plus preview images."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import build_truck as bt  # noqa: E402
from render import render, VIEWS  # noqa: E402

SKIPS = {
    "bobs_daily": {"bt_plowblade"},
    "bobs_plow": set(),
    "bobs_clean": {"bt_plowblade", "bt_plowmount", "bt_toolbox"},
}

if __name__ == "__main__":
    objs = bt.build_all()
    eye, tgt = VIEWS["front_34"]
    for cfg, skip in SKIPS.items():
        im = render(objs, eye, tgt, W=500, H=281, fov=30, skip=skip)
        im.convert("RGB").save(os.path.join(bt.VEH, f"{cfg}.jpg"), quality=90)
        if cfg == "bobs_daily":
            im.convert("RGB").save(os.path.join(bt.VEH, "default.jpg"), quality=90)
    out = os.path.join(bt.ROOT, "preview")
    os.makedirs(out, exist_ok=True)
    for name, (eye, tgt) in VIEWS.items():
        render(objs, eye, tgt).save(os.path.join(out, f"{name}.png"))
    render(objs, *VIEWS["front_34"], skip=set()).save(os.path.join(out, "front_34_plow.png"))
    print("thumbnails + previews written")
