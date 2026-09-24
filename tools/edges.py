"""Draw the model's edges (cyan) over each reference photo from its solved camera: preview/edges_<photo>.jpg"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(__file__))
import build_truck as bt  # noqa: E402
from camfit import camera_from_params  # noqa: E402
from render import render_cam  # noqa: E402

if __name__ == "__main__":
    cams = json.load(open(os.path.join(bt.ROOT, "tools", "cameras.json")))
    only = sys.argv[1:] or list(cams)
    objs = bt.build_all()
    for name in only:
        size = (1600, 1200)
        photo = Image.open(os.path.join(bt.ROOT, "reference", f"{name}.jpg")).convert("RGB").resize(size)
        ren, mask = render_cam(objs, camera_from_params(cams[name]), size, skip={"bt_plowblade"})
        e = np.asarray(ren.convert("L").filter(ImageFilter.FIND_EDGES), np.float32)
        e = np.maximum(e, np.asarray(mask.filter(ImageFilter.FIND_EDGES), np.float32))
        e = (e > 18).astype(np.uint8) * 255
        edge = Image.fromarray(e).filter(ImageFilter.MaxFilter(3))
        out = photo.copy()
        out.paste(Image.new("RGB", size, (0, 255, 255)), (0, 0), edge)
        out.save(os.path.join(bt.ROOT, "preview", f"edges_{name}.jpg"), quality=88)
        print("wrote", name)
