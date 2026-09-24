"""Static consistency checks for the mod folder (run after building)."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from build_truck import VEH  # noqa: E402

errors, warnings = [], []


def load_jbeams():
    parts = {}
    for fn in sorted(os.listdir(VEH)):
        if fn.endswith(".jbeam"):
            with open(os.path.join(VEH, fn)) as f:
                data = json.load(f)
            for k, v in data.items():
                if k in parts:
                    errors.append(f"duplicate part {k}")
                parts[k] = v
    return parts


def rows(section):
    return [r for r in section[1:] if isinstance(r, list)]


def main():
    parts = load_jbeams()
    dae = open(os.path.join(VEH, "bobs_truck.dae")).read()
    dae_objects = set(re.findall(r'<geometry id="[^"]+" name="([^"]+)"', dae))
    dae_mats = set(re.findall(r'<material id="[^"]+" name="([^"]+)"', dae))
    mats = json.load(open(os.path.join(VEH, "main.materials.json")))

    for m in sorted(dae_mats - set(mats)):
        errors.append(f"material {m} used in dae but not defined")
    for name, m in mats.items():
        tex = m["Stages"][0].get("baseColorMap")
        if tex and not os.path.exists(os.path.join(VEH, os.path.basename(tex))):
            errors.append(f"missing texture {tex}")

    node_ids, node_groups = {}, {}
    for pname, p in parts.items():
        grp = ""
        for r in p.get("nodes", [])[1:]:
            if isinstance(r, dict):
                grp = r.get("group", grp)
                continue
            if r[0] in node_ids:
                errors.append(f"duplicate node {r[0]} ({pname}, {node_ids[r[0]]})")
            node_ids[r[0]] = pname
            for g in ([grp] if isinstance(grp, str) else grp):
                if g:
                    node_groups.setdefault(g, 0)
                    node_groups[g] += 1

    wheel_groups = set()
    for p in parts.values():
        for r in rows(p.get("pressureWheels", [[]])):
            wheel_groups |= {r[1], r[2]}
            for n in (r[3], r[4], r[6]):
                if n not in node_ids:
                    errors.append(f"wheel {r[0]} references missing node {n}")

    degree = {n: 0 for n in node_ids}
    for pname, p in parts.items():
        for sec in ("beams", "hydros"):
            for r in rows(p.get(sec, [[]])):
                for n in r[:2]:
                    if n not in node_ids:
                        errors.append(f"{pname}.{sec} references missing node {n}")
                    else:
                        degree[n] += 1
        for r in rows(p.get("triangles", [[]])):
            for n in r[:3]:
                if n not in node_ids:
                    errors.append(f"{pname}.triangles references missing node {n}")
        for r in rows(p.get("flexbodies", [[]])):
            mesh, groups = r[0], r[1]
            if mesh not in dae_objects:
                errors.append(f"flexbody mesh {mesh} not in dae")
            for g in groups:
                if g not in node_groups and g not in wheel_groups:
                    errors.append(f"flexbody {mesh} uses unknown group {g}")
        cams = p.get("camerasInternal")
        if cams:
            for r in rows(cams):
                for n in r[5:11]:
                    if n not in node_ids:
                        errors.append(f"camera references missing node {n}")
        ref = p.get("refNodes")
        if ref:
            for n in ref[1]:
                if n not in node_ids:
                    errors.append(f"refNode {n} missing")
    for n, d in degree.items():
        if d < 3:
            errors.append(f"node {n} only has {d} beams (unstable)")

    # slots / configs
    slot_types = {p.get("slotType") for p in parts.values()}
    for fn in os.listdir(VEH):
        if fn.endswith(".pc"):
            pc = json.load(open(os.path.join(VEH, fn)))
            for slot, part in pc["parts"].items():
                if slot not in slot_types:
                    errors.append(f"{fn}: unknown slot {slot}")
                if part and part not in parts:
                    errors.append(f"{fn}: unknown part {part}")
    used_meshes = {r[0] for p in parts.values() for r in rows(p.get("flexbodies", [[]]))}
    for m in sorted(dae_objects - used_meshes):
        warnings.append(f"dae object {m} not used by any flexbody")

    print(f"{len(parts)} parts, {len(node_ids)} nodes, {len(dae_objects)} meshes, {len(mats)} materials")
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("VALID" if not errors else f"{len(errors)} errors")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
