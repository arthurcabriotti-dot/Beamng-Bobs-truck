"""Generates the BeamNG physics (jbeam), configs and info files for Bob's K20.

Structure:
  bobs_truck.jbeam            main part: frame, body, solid axles, steering, wheels, engine, driveline
  bobs_truck_accessories.jbeam  toolbox, Western plow mount, plow blade
  *.pc / info_*.json          vehicle configurations
  info.json                   model info for the vehicle selector
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from build_truck import VEH, WB, TIRE_R, TRACK_F, TRACK_R, HALF_W  # noqa: E402

FLT_MAX = "FLT_MAX"
G = 9.81

nodes = {}        # id -> dict(pos, group, weight, extra)
node_order = []


def node(nid, x, y, z, group, weight, **extra):
    assert nid not in nodes, nid
    nodes[nid] = {"pos": (round(x, 4), round(y, 4), round(z, 4)), "group": group, "w": weight, "extra": extra}
    node_order.append(nid)
    return nid


def dist(a, b):
    pa, pb = nodes[a]["pos"], nodes[b]["pos"]
    return math.dist(pa, pb)


# =============================================================== nodes
# ---- frame ladder
FRAME_Y = [-0.86, -0.45, 0.0, 0.6, 1.2, 1.9, 2.6, WB, 4.0, 4.46]
frame = []
for i, y in enumerate(FRAME_Y):
    for s, sn in ((1, "l"), (-1, "r")):
        for z, zn in ((0.48, "b"), (0.68, "t")):
            frame.append(node(f"fr{i}{sn}{zn}", s * 0.44, y, z, "bt_frame", 12.0))
# spring towers (rigidly tied to the frame)
towers = []
for yname, y in (("f", 0.0), ("r", WB)):
    for s, sn in ((1, "l"), (-1, "r")):
        towers.append(node(f"{yname}st{sn}", s * 0.46, y, 0.98, "bt_frame", 5.0))


# ---- body shells
def lattice(prefix, group, ys, xs, zs, w):
    out = []
    for i, y in enumerate(ys):
        for x, sn in xs:
            for z, zn in zs:
                out.append(node(f"{prefix}{i}{sn}{zn}", x, y, z, group, w))
    return out


front = lattice("bf", "bt_front", [-0.80, -0.25, 0.35], [(0.95, "l"), (-0.95, "r")],
                [(0.72, "b"), (1.30, "t")], 9.0)
cab = lattice("bc", "bt_cab", [0.74, 1.35, 1.96], [(0.98, "l"), (-0.98, "r")],
              [(0.66, "b"), (1.38, "t")], 18.0)
cab += [node("bcr0l", 0.80, 1.17, 1.85, "bt_cab", 10.0), node("bcr0r", -0.80, 1.17, 1.85, "bt_cab", 10.0),
        node("bcr1l", 0.80, 1.94, 1.85, "bt_cab", 10.0), node("bcr1r", -0.80, 1.94, 1.85, "bt_cab", 10.0)]
bed = lattice("bb", "bt_bed", [2.03, 2.80, 3.60, 4.42], [(0.97, "l"), (-0.97, "r")],
              [(0.68, "b"), (1.40, "t")], 14.0)

# ---- engine / transmission
engine = []
for i, y in ((1, -0.50), (2, 0.25)):
    for s, sn in ((1, "l"), (-1, "r")):
        engine.append(node(f"e{i}{sn}b", s * 0.28, y, 0.62, "bt_engine", 40.0))
        engine.append(node(f"e{i}{sn}t", s * 0.28, y, 1.05, "bt_engine", 40.0))
engine.append(node("tr1", 0.0, 1.00, 0.66, "bt_engine", 60.0))

# ---- front solid axle
faxle = [
    node("fkptl", 0.78, 0.0, TIRE_R + 0.135, "bt_faxle", 18.0), node("fkpbl", 0.80, 0.0, TIRE_R - 0.135, "bt_faxle", 18.0),
    node("fkptr", -0.78, 0.0, TIRE_R + 0.135, "bt_faxle", 18.0), node("fkpbr", -0.80, 0.0, TIRE_R - 0.135, "bt_faxle", 18.0),
    node("fapl", 0.46, 0.0, 0.50, "bt_faxle", 18.0), node("fapr", -0.46, 0.0, 0.50, "bt_faxle", 18.0),
    node("fall", 0.46, 0.0, 0.30, "bt_faxle", 18.0), node("falr", -0.46, 0.0, 0.30, "bt_faxle", 18.0),
    node("fac1", 0.20, 0.14, 0.40, "bt_faxle", 18.0), node("fac2", 0.0, -0.14, 0.34, "bt_faxle", 18.0),
]
fhub = {}
for s, sn, W in ((1, "l", "FL"), (-1, "r", "FR")):
    fhub[sn] = [node(f"fw1{sn}", s * 0.70, 0.0, TIRE_R, "wheel_" + W, 8.0),
                node(f"fw2{sn}", s * 0.98, 0.0, TIRE_R, "wheel_" + W, 8.0),
                node(f"fsa{sn}", s * 0.755, 0.20, 0.42, "bt_faxle", 6.0)]

# ---- rear solid axle (14-bolt full floater)
raxle = [
    node("rael", 0.74, WB, TIRE_R + 0.10, "bt_raxle", 20.0), node("raer", -0.74, WB, TIRE_R + 0.10, "bt_raxle", 20.0),
    node("rapl", 0.46, WB, 0.50, "bt_raxle", 22.0), node("rapr", -0.46, WB, 0.50, "bt_raxle", 22.0),
    node("rall", 0.46, WB, 0.30, "bt_raxle", 22.0), node("ralr", -0.46, WB, 0.30, "bt_raxle", 22.0),
    node("rac1", 0.0, WB + 0.16, 0.42, "bt_raxle", 24.0), node("rac2", 0.0, WB - 0.16, 0.34, "bt_raxle", 24.0),
]
rhub = {}
for s, sn, W in ((1, "l", "RL"), (-1, "r", "RR")):
    rhub[sn] = [node(f"rw1{sn}", s * 0.70, WB, TIRE_R, "wheel_" + W, 10.0),
                node(f"rw2{sn}", s * 0.96, WB, TIRE_R, "wheel_" + W, 10.0)]

# =============================================================== beams
beams = []   # (a, b, props-key)
BEAM_PROPS = {
    "frame":  {"beamSpring": 12001000, "beamDamp": 350, "beamDeform": 280000, "beamStrength": FLT_MAX},
    "body":   {"beamSpring": 3001000, "beamDamp": 120, "beamDeform": 60000, "beamStrength": FLT_MAX},
    "mount":  {"beamSpring": 2001000, "beamDamp": 150, "beamDeform": 90000, "beamStrength": FLT_MAX},
    "engine": {"beamSpring": 12001000, "beamDamp": 300, "beamDeform": FLT_MAX, "beamStrength": FLT_MAX},
    "rigid":  {"beamSpring": 10001000, "beamDamp": 300, "beamDeform": FLT_MAX, "beamStrength": FLT_MAX},
    "link":   {"beamSpring": 6001000, "beamDamp": 250, "beamDeform": FLT_MAX, "beamStrength": FLT_MAX},
}
seen = set()


def beam(a, b, kind):
    key = tuple(sorted((a, b)))
    if a == b or key in seen:
        return
    seen.add(key)
    beams.append((a, b, kind))


def mesh_group(ids, kind, maxd=None):
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if maxd is None or dist(a, b) <= maxd:
                beam(a, b, kind)


def anchor(ids, targets, k, kind):
    for a in ids:
        near = sorted(targets, key=lambda t: dist(a, t))[:k]
        for t in near:
            beam(a, t, kind)


mesh_group(frame, "frame", maxd=1.25)
anchor(towers, frame, 6, "frame")
beam("fstl", "fstr", "frame"); beam("rstl", "rstr", "frame")
mesh_group(front, "body")
mesh_group(cab, "body")
mesh_group(bed, "body")
anchor([n for n in front if n.endswith("b")], frame, 3, "mount")
anchor(front, cab, 2, "mount")
anchor(front, towers[:2], 1, "mount")
anchor([n for n in cab if n.endswith("b")], frame, 3, "mount")
anchor([n for n in bed if n.endswith("b")], frame, 3, "mount")
mesh_group(engine, "engine")
anchor(engine, frame, 3, "mount")

# axles: rigid bodies
mesh_group(faxle, "rigid")
mesh_group(raxle + rhub["l"] + rhub["r"], "rigid")
for sn in "lr":
    fw1, fw2, fsa = fhub[sn]
    mesh_group([fw1, fw2, fsa], "rigid")
    for h in (fw1, fw2, fsa):  # rotate about the king pin
        beam(h, f"fkpt{sn}", "rigid")
        beam(h, f"fkpb{sn}", "rigid")
beam("fsal", "fsar", "rigid")  # tie rod

# 4-link + panhard locate each axle (stands in for the leaf springs' location duty)
for sn in "lr":
    beam(f"fal{sn}", f"fr3{sn}b", "link")
    beam(f"fap{sn}", f"fr3{sn}t", "link")
    beam(f"ral{sn}", f"fr6{sn}b", "link")
    beam(f"rap{sn}", f"fr6{sn}t", "link")
beam("fapr", "fr2lb", "link")
beam("rapr", "fr7lb", "link")

# ---- spring / damper beams, rates from static corner loads
axle_groups = {"bt_faxle", "bt_raxle", "wheel_FL", "wheel_FR", "wheel_RL", "wheel_RR"}
sprung = [n for n in node_order if nodes[n]["group"] not in axle_groups]
m_sprung = sum(nodes[n]["w"] for n in sprung)
y_cg = sum(nodes[n]["w"] * nodes[n]["pos"][1] for n in sprung) / m_sprung
m_front = m_sprung * (WB - y_cg) / WB
m_rear = m_sprung - m_front


def spring_props(m_corner, freq, zeta, L):
    w = 2 * math.pi * freq
    k = m_corner * w * w
    c = 2 * zeta * math.sqrt(k * m_corner)
    pre = 1 + (m_corner * G) / (k * L)
    return {"beamSpring": round(k), "beamDamp": round(c), "beamPrecompression": round(pre, 4),
            "beamDeform": FLT_MAX, "beamStrength": FLT_MAX}


L_spring = 0.98 - 0.50
SPRING_F = spring_props(m_front / 2, 1.8, 0.35, L_spring)
SPRING_R = spring_props(m_rear / 2, 1.9, 0.35, L_spring)
springs = [("fapl", "fstl", "F"), ("fapr", "fstr", "F"), ("rapl", "rstl", "R"), ("rapr", "rstr", "R")]

# ---- collision triangles (outer hull of each body shell)
triangles = []


def tri_quad(a, b, c, d, center):
    """Two triangles for quad a,b,c,d oriented to face away from `center`."""
    pa, pb, pc = (nodes[x]["pos"] for x in (a, b, c))
    n = cross3(sub3(pb, pa), sub3(pc, pa))
    mid = [(nodes[a]["pos"][i] + nodes[c]["pos"][i]) / 2 for i in range(3)]
    if sum(n[i] * (mid[i] - center[i]) for i in range(3)) < 0:
        a, b, c, d = d, c, b, a
    triangles.append((a, b, c))
    triangles.append((a, c, d))


def sub3(a, b): return [a[i] - b[i] for i in range(3)]
def cross3(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def hull_box(prefix, n_st, center, top=True, front_face=True, back_face=True):
    for i in range(n_st - 1):
        for sn in "lr":
            tri_quad(f"{prefix}{i}{sn}b", f"{prefix}{i + 1}{sn}b", f"{prefix}{i + 1}{sn}t", f"{prefix}{i}{sn}t", center)
        if top:
            tri_quad(f"{prefix}{i}lt", f"{prefix}{i + 1}lt", f"{prefix}{i + 1}rt", f"{prefix}{i}rt", center)
    if front_face:
        tri_quad(f"{prefix}0lb", f"{prefix}0rb", f"{prefix}0rt", f"{prefix}0lt", center)
    if back_face:
        j = n_st - 1
        tri_quad(f"{prefix}{j}lb", f"{prefix}{j}rb", f"{prefix}{j}rt", f"{prefix}{j}lt", center)


hull_box("bf", 3, (0, 0.0, 1.0), back_face=False)
hull_box("bc", 3, (0, 1.35, 1.1), top=False, front_face=False)
tri_quad("bcr0l", "bcr1l", "bcr1r", "bcr0r", (0, 1.5, 1.2))              # roof
tri_quad("bc0lt", "bc0rt", "bcr0r", "bcr0l", (0, 1.5, 1.2))              # windshield
tri_quad("bc2lt", "bc2rt", "bcr1r", "bcr1l", (0, 1.2, 1.2))              # back glass
hull_box("bb", 4, (0, 3.2, 1.0), top=False)


# =============================================================== assemble jbeam
def node_rows(ids, default_props):
    rows = [["id", "posX", "posY", "posZ"], dict(default_props)]
    cur_group, cur_w = None, None
    for n in ids:
        d = nodes[n]
        props = {}
        if d["group"] != cur_group:
            props["group"] = d["group"]
            cur_group = d["group"]
        if d["w"] != cur_w:
            props["nodeWeight"] = d["w"]
            cur_w = d["w"]
        if props:
            rows.append(props)
        rows.append([n, *d["pos"]])
    rows.append({"group": "", "nodeWeight": 10})
    return rows


def beam_rows():
    rows = [["id1:", "id2:"]]
    cur = None
    for a, b, kind in beams:
        if kind != cur:
            rows.append(dict(BEAM_PROPS[kind]))
            cur = kind
        rows.append([a, b])
    rows.append(dict(SPRING_F))
    for a, b, ax in springs:
        if ax == "F":
            rows.append([a, b])
    rows.append(dict(SPRING_R))
    for a, b, ax in springs:
        if ax == "R":
            rows.append([a, b])
    rows.append({"beamPrecompression": 1, "beamSpring": 3001000, "beamDamp": 150,
                 "beamDeform": 60000, "beamStrength": FLT_MAX})
    return rows


TIRE = {
    "radius": TIRE_R, "tireWidth": 0.25, "hubRadius": 0.215, "hubWidth": 0.18,
    "numRays": 20, "hasTire": True, "enableHubcaps": False,
    "enableTireLbeams": True, "enableTireSideReinfBeams": True, "enableTireReinfBeams": False,
    "enableTreadReinfBeams": True, "enableTirePeripheryReinfBeams": True,
    "pressurePSI": 45, "nodeWeight": 0.5, "hubNodeWeight": 0.6,
    "nodeMaterial": "|NM_RUBBER", "hubNodeMaterial": "|NM_METAL",
    "frictionCoef": 1.0, "slidingFrictionCoef": 0.72, "treadCoef": 1.0,
    "noLoadCoef": 1.45, "fullLoadCoef": 0.5, "loadSensitivitySlope": 0.00019, "softnessCoef": 0.6,
    "stribeckExponent": 1.75, "stribeckVelMult": 1.0,
    "wheelSideBeamSpring": 12000, "wheelSideBeamDamp": 45, "wheelSideBeamDeform": FLT_MAX, "wheelSideBeamStrength": FLT_MAX,
    "wheelSideTransverseBeamSpring": 18000, "wheelSideTransverseBeamDamp": 40,
    "wheelSideReinfBeamSpring": 3500, "wheelSideReinfBeamDamp": 15,
    "wheelReinfBeamSpring": 1800, "wheelReinfBeamDamp": 20,
    "wheelTreadBeamSpring": 350000, "wheelTreadBeamDamp": 60, "wheelTreadBeamDeform": FLT_MAX, "wheelTreadBeamStrength": FLT_MAX,
    "wheelTreadReinfBeamSpring": 8000, "wheelTreadReinfBeamDamp": 30,
    "wheelPeripheryBeamSpring": 1500000, "wheelPeripheryBeamDamp": 60, "wheelPeripheryBeamDeform": FLT_MAX, "wheelPeripheryBeamStrength": FLT_MAX,
    "wheelPeripheryReinfBeamSpring": 30000, "wheelPeripheryReinfBeamDamp": 30,
    "hubBeamSpring": 1500000, "hubBeamDamp": 80, "hubBeamDeform": FLT_MAX, "hubBeamStrength": FLT_MAX,
    "hubTreadBeamSpring": 1500000, "hubTreadBeamDamp": 80,
    "hubPeripheryBeamSpring": 1500000, "hubPeripheryBeamDamp": 80,
    "hubSideBeamSpring": 1500000, "hubSideBeamDamp": 80,
    "hubReinfBeamSpring": 1500000, "hubReinfBeamDamp": 80,
    "hubStabilizerBeamSpring": 1500000, "hubStabilizerBeamDamp": 80,
    "dragCoef": 5, "triangleCollision": False, "selfCollision": False, "collision": True,
    "enableABS": False, "brakeDiameter": 0.30, "brakeMass": 12, "brakeType": "disc",
}


def pressure_wheels():
    rows = [["name", "hubGroup", "group", "node1:", "node2:", "nodeS", "nodeArm:", "wheelDir"], dict(TIRE)]
    rows.append({"brakeTorque": 2800, "parkingTorque": 0})
    rows.append(["FR", "wheel_FR", "tire_FR", "fw1r", "fw2r", 9999, "fsar", 1, {"speedo": True}])
    rows.append(["FL", "wheel_FL", "tire_FL", "fw1l", "fw2l", 9999, "fsal", -1])
    rows.append({"brakeTorque": 1900, "parkingTorque": 2200, "brakeType": "drum"})
    rows.append(["RR", "wheel_RR", "tire_RR", "rw1r", "rw2r", 9999, "rapr", 1])
    rows.append(["RL", "wheel_RL", "tire_RL", "rw1l", "rw2l", 9999, "rapl", -1])
    rows.append({"selfCollision": True, "collision": True})
    return rows


FLEX = [
    ("bt_frame_mesh", ["bt_frame"]),
    ("bt_bumper_F", ["bt_frame"]),
    ("bt_bumper_R", ["bt_frame"]),
    ("bt_engine_mesh", ["bt_engine", "bt_frame"]),
    ("bt_springs", ["bt_frame", "bt_faxle", "bt_raxle"]),
    ("bt_axle_F", ["bt_faxle"]),
    ("bt_axle_R", ["bt_raxle"]),
    ("bt_body_front", ["bt_front"]),
    ("bt_body_cab", ["bt_cab"]),
    ("bt_glass", ["bt_cab"]),
    ("bt_bed", ["bt_bed"]),
    ("bt_tailgate", ["bt_bed"]),
]
for W in ("FL", "FR", "RL", "RR"):
    FLEX.append((f"bt_wheel_{W}", [f"wheel_{W}"]))
    FLEX.append((f"bt_tire_{W}", [f"tire_{W}"]))

# 350 4bbl small block (~165 hp / 275 lb-ft)
TORQUE = [[0, 0], [500, 250], [1000, 330], [1600, 372], [2000, 370], [2400, 362], [2800, 350],
          [3200, 335], [3600, 312], [4000, 285], [4400, 245], [4800, 200]]

main = {
    "information": {"authors": "Bob's Truck mod", "name": "1985 Chevrolet K20 (Bob's Truck)", "value": 9500},
    "slotType": "main",
    "slots": [
        ["type", "default", "description"],
        ["bt_toolbox", "bt_toolbox_uws", "Bed Toolbox"],
        ["bt_plowmount", "bt_plowmount_western", "Snow Plow Mount"],
    ],
    "refNodes": [["ref:", "back:", "left:", "up:"], ["fr4rb", "fr5rb", "fr4lb", "fr4rt"]],
    "cameraExternal": {"distance": 7.5, "distanceMin": 4, "offset": {"x": 0, "y": 0.3, "z": 0.9}, "fov": 65},
    "camerasInternal": [
        ["type", "x", "y", "z", "fov", "id1:", "id2:", "id3:", "id4:", "id5:", "id6:"],
        {"nodeWeight": 1.3, "selfCollision": False, "collision": False},
        ["dash", 0.40, 1.55, 1.62, 65, "bc1lt", "bc1rt", "bc2lt", "bcr0l", "bcr1l", "bc1lb"],
    ],
    "flexbodies": [["mesh", "[group]:", "nonFlexMaterials"]] + [[m, g] for m, g in FLEX],
    "nodes": node_rows(node_order, {"frictionCoef": 0.7, "nodeMaterial": "|NM_METAL", "collision": True,
                                    "selfCollision": True}),
    "beams": beam_rows(),
    "triangles": [["id1:", "id2:", "id3:"], {"groundModel": "metal"}] + [list(t) for t in triangles],
    "hydros": [
        ["id1:", "id2:"],
        {"beamSpring": 6001000, "beamDamp": 250, "beamDeform": FLT_MAX, "beamStrength": FLT_MAX},
        ["fapr", "fsal", {"inputSource": "steering_input", "inputFactor": 1, "factor": 0.082,
                          "inRate": 2.2, "outRate": 2.2, "steeringWheelLock": 540}],
    ],
    "pressureWheels": pressure_wheels(),
    "powertrain": [
        ["type", "name", "inputName", "inputIndex"],
        ["combustionEngine", "mainEngine", "dummy", 0],
        ["torqueConverter", "torqueConverter", "mainEngine", 1],
        ["automaticGearbox", "gearbox", "torqueConverter", 1],
        ["differential", "transfercase", "gearbox", 1, {"diffType": "locked", "uiName": "Transfer Case (4 High)"}],
        ["differential", "differential_R", "transfercase", 1, {"deviceCategories": {"differential": True}}],
        ["differential", "differential_F", "transfercase", 2, {"deviceCategories": {"differential": True}}],
        ["shaft", "wheelaxleRL", "differential_R", 1, {"connectedWheel": "RL"}],
        ["shaft", "wheelaxleRR", "differential_R", 2, {"connectedWheel": "RR"}],
        ["shaft", "wheelaxleFL", "differential_F", 1, {"connectedWheel": "FL"}],
        ["shaft", "wheelaxleFR", "differential_F", 2, {"connectedWheel": "FR"}],
    ],
    "mainEngine": {
        "torque": [["rpm", "torque"]] + TORQUE,
        "idleRPM": 650, "maxRPM": 4700, "revLimiterRPM": 4600, "revLimiterType": "soft",
        "inertia": 0.25, "friction": 18, "dynamicFriction": 0.024, "engineBrakeTorque": 45,
        "burnEfficiency": [[0, 0.12], [0.05, 0.22], [0.4, 0.28], [0.7, 0.30], [1, 0.26]],
        "energyStorage": "mainTank", "requiredEnergyType": "gasoline",
        "torqueReactionNodes:": ["e1lb", "e1rb", "e2lt"],
        "starterTorque": 120, "starterMaxAV": 700,
        "soundConfig": "soundConfig",
        "uiName": "350 V8",
    },
    "soundConfig": {
        "sampleName": "V8_2_engine",
        "intakeMuffling": 0.9, "mainGain": -2, "onLoadGain": 1, "offLoadGain": 0.55,
        "maxLoadMix": 0.7, "minLoadMix": 0,
        "lowShelfGain": 2, "lowShelfFreq": 90, "highShelfGain": -1, "highShelfFreq": 2500,
        "eqLowGain": 0, "eqLowFreq": 200, "eqLowWidth": 0.2,
        "eqHighGain": 0, "eqHighFreq": 2000, "eqHighWidth": 0.2,
        "fundamentalFrequencyCylinderCount": 8, "eqFundamentalGain": -4,
    },
    "torqueConverter": {
        "uiName": "Torque Converter", "converterDiameter": 0.30, "converterStiffness": 9,
        "couplingAVRatio": 0.9, "stallTorqueRatio": 2.0, "lockupClutchTorque": 0,
    },
    # TH400 3-speed automatic
    "gearbox": {
        "uiName": "TH400 3-Speed Automatic",
        "gearRatios": [-2.08, 0, 2.48, 1.48, 1.00],
        "parkLockTorque": 4000, "friction": 1.2, "dynamicFriction": 0.0012, "torqueLossCoef": 0.02,
    },
    "transfercase": {"gearRatio": 1.0, "friction": 1.0, "dynamicFriction": 0.0008, "torqueLossCoef": 0.015},
    "differential_R": {"diffType": "open", "gearRatio": 4.10, "friction": 2, "dynamicFriction": 0.001,
                       "torqueLossCoef": 0.02, "uiName": "Rear Differential (14-bolt)"},
    "differential_F": {"diffType": "open", "gearRatio": 4.10, "friction": 2, "dynamicFriction": 0.001,
                       "torqueLossCoef": 0.02, "uiName": "Front Differential"},
    "energyStorage": [["type", "name"], ["fuelTank", "mainTank"]],
    "mainTank": {"energyType": "gasoline", "fuelCapacity": 76, "startingFuelCapacity": 50,
                 "fuel": {"[engineGroup]:": ["fuel"]}},
    "vehicleController": {"shiftDownRPMOffsetCoef": 1.15, "aggressionSmoothingUp": 1.5},
    "controller": [["fileName"], ["vehicleController", {}]],
    "input": {"steeringWheelLock": 540},
}
main["mainTank"].pop("fuel")

ACC = {
    "bt_toolbox_uws": {
        "information": {"authors": "Bob's Truck mod", "name": "UWS Crossover Toolbox", "value": 350},
        "slotType": "bt_toolbox",
        "flexbodies": [["mesh", "[group]:", "nonFlexMaterials"], ["bt_toolbox", ["bt_bed"]]],
        "nodes": [["id", "posX", "posY", "posZ"], {"group": "bt_bed", "nodeWeight": 6.0},
                  ["tb1l", 0.95, 2.32, 1.55], ["tb1r", -0.95, 2.32, 1.55], {"group": ""}],
        "beams": [["id1:", "id2:"],
                  {"beamSpring": 1001000, "beamDamp": 80, "beamDeform": 30000, "beamStrength": FLT_MAX},
                  ["tb1l", "tb1r"], ["tb1l", "bb0lt"], ["tb1l", "bb1lt"], ["tb1l", "bb0rt"], ["tb1l", "bb0lb"],
                  ["tb1r", "bb0rt"], ["tb1r", "bb1rt"], ["tb1r", "bb0lt"], ["tb1r", "bb0rb"]],
    },
    "bt_plowmount_western": {
        "information": {"authors": "Bob's Truck mod", "name": "Western Unimount Plow Frame + Lights", "value": 900},
        "slotType": "bt_plowmount",
        "slots": [["type", "default", "description"], ["bt_plow", "", "Plow Blade"]],
        "flexbodies": [["mesh", "[group]:", "nonFlexMaterials"], ["bt_plowmount", ["bt_plowmount"]]],
        "nodes": [["id", "posX", "posY", "posZ"], {"group": "bt_plowmount", "nodeWeight": 16.0},
                  ["pm1l", 0.42, -1.14, 0.42], ["pm1r", -0.42, -1.14, 0.42],
                  ["pm2l", 0.36, -1.13, 1.36], ["pm2r", -0.36, -1.13, 1.36],
                  ["pm3", 0.0, -1.20, 0.40], {"group": ""}],
        "beams": [["id1:", "id2:"],
                  {"beamSpring": 6001000, "beamDamp": 250, "beamDeform": 150000, "beamStrength": FLT_MAX},
                  ["pm1l", "pm1r"], ["pm2l", "pm2r"], ["pm1l", "pm2l"], ["pm1r", "pm2r"], ["pm1l", "pm2r"],
                  ["pm1r", "pm2l"], ["pm3", "pm1l"], ["pm3", "pm1r"], ["pm3", "pm2l"], ["pm3", "pm2r"],
                  ["pm1l", "fr0lb"], ["pm1l", "fr0lt"], ["pm1l", "fr1lb"], ["pm1r", "fr0rb"], ["pm1r", "fr0rt"],
                  ["pm1r", "fr1rb"], ["pm3", "fr0lb"], ["pm3", "fr0rb"],
                  ["pm2l", "fr0lt"], ["pm2r", "fr0rt"], ["pm2l", "bf0lt"], ["pm2r", "bf0rt"],
                  ["pm2l", "fr1lt"], ["pm2r", "fr1rt"]],
        "triangles": [["id1:", "id2:", "id3:"], {"groundModel": "metal"},
                      ["pm1l", "pm1r", "pm2r"], ["pm1l", "pm2r", "pm2l"]],
    },
    "bt_plow_western": {
        "information": {"authors": "Bob's Truck mod", "name": "Western 7.5' Straight Blade (raised)", "value": 1800},
        "slotType": "bt_plow",
        "flexbodies": [["mesh", "[group]:", "nonFlexMaterials"], ["bt_plowblade", ["bt_plow"]]],
        "nodes": [["id", "posX", "posY", "posZ"], {"group": "bt_plow", "nodeWeight": 30.0},
                  ["pb1l", 1.14, -1.42, 0.18], ["pb1r", -1.14, -1.42, 0.18],
                  ["pb2l", 1.14, -1.40, 0.88], ["pb2r", -1.14, -1.40, 0.88],
                  ["pb3l", 0.50, -1.28, 0.45], ["pb3r", -0.50, -1.28, 0.45],
                  ["pb4", 0.0, -1.44, 0.18], {"group": ""}],
        "beams": [["id1:", "id2:"],
                  {"beamSpring": 8001000, "beamDamp": 300, "beamDeform": 200000, "beamStrength": FLT_MAX},
                  ["pb1l", "pb1r"], ["pb2l", "pb2r"], ["pb1l", "pb2l"], ["pb1r", "pb2r"], ["pb1l", "pb2r"],
                  ["pb1r", "pb2l"], ["pb3l", "pb3r"], ["pb3l", "pb1l"], ["pb3l", "pb2l"], ["pb3r", "pb1r"],
                  ["pb3r", "pb2r"], ["pb3l", "pb2r"], ["pb3r", "pb2l"], ["pb4", "pb1l"], ["pb4", "pb1r"],
                  ["pb4", "pb3l"], ["pb4", "pb3r"], ["pb4", "pb2l"], ["pb4", "pb2r"],
                  ["pb3l", "pm1l"], ["pb3l", "pm2l"], ["pb3l", "pm3"], ["pb3r", "pm1r"], ["pb3r", "pm2r"],
                  ["pb3r", "pm3"], ["pb4", "pm3"], ["pb4", "pm1l"], ["pb4", "pm1r"], ["pb2l", "pm2l"],
                  ["pb2r", "pm2r"], ["pb1l", "pm1l"], ["pb1r", "pm1r"]],
        "triangles": [["id1:", "id2:", "id3:"], {"groundModel": "metal"},
                      ["pb1r", "pb1l", "pb2l"], ["pb1r", "pb2l", "pb2r"]],
    },
}

# =============================================================== configs + info
PAINT = {"baseColor": [0.62, 0.02, 0.025, 1.2], "metallic": 0.05, "roughness": 0.35,
         "clearcoat": 0.5, "clearcoatRoughness": 0.2}
CONFIGS = {
    "bobs_daily": ({"bt_toolbox": "bt_toolbox_uws", "bt_plowmount": "bt_plowmount_western", "bt_plow": ""},
                   {"Configuration": "Bob's Truck (Plow Frame)", "Value": 9500,
                    "Description": "Bob's 1985 K20 as it sits in the driveway: plow frame and lights on, blade off."}),
    "bobs_plow": ({"bt_toolbox": "bt_toolbox_uws", "bt_plowmount": "bt_plowmount_western", "bt_plow": "bt_plow_western"},
                  {"Configuration": "Bob's Truck (Plowing)", "Value": 11300,
                   "Description": "Western straight blade hung on the front, ready for a Connecticut snowstorm."}),
    "bobs_clean": ({"bt_toolbox": "", "bt_plowmount": "", "bt_plow": ""},
                   {"Configuration": "Bob's Truck (Summer)", "Value": 8500,
                    "Description": "Plow frame and toolbox removed."}),
}


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")


def main_():
    write_json(os.path.join(VEH, "bobs_truck.jbeam"), {"bobs_truck": main})
    write_json(os.path.join(VEH, "bobs_truck_accessories.jbeam"), ACC)
    for name, (parts, info) in CONFIGS.items():
        write_json(os.path.join(VEH, f"{name}.pc"),
                   {"format": 2, "model": "bobs_truck", "parts": parts, "vars": {}, "paints": [PAINT]})
        write_json(os.path.join(VEH, f"info_{name}.json"),
                   dict(info, **{"Drivetrain": "4WD", "Transmission": "Automatic", "Fuel Type": "Gasoline",
                                 "Power": 167, "Torque": 372, "Weight": round(total_mass()), "Config Type": "Factory"}))
    write_json(os.path.join(VEH, "info.json"), {
        "Name": "K20 (Bob's Truck)", "Brand": "Chevrolet", "Type": "Truck", "Body Style": "Pickup",
        "Country": "United States", "Derby Class": "Heavy Truck", "Years": {"min": 1985, "max": 1986},
        "Region": "North America",
        "Description": "Bob's 1985 Chevrolet K20 square body: regular cab long bed, 350 V8, TH400, 4x4, "
                       "8-lug steelies, chrome step bumper, UWS toolbox and a Western plow setup.",
        "Author": "Bob's Truck mod", "default_pc": "bobs_daily",
        "paints": {"Cardinal Red (faded)": PAINT,
                   "Frost White": {"baseColor": [0.85, 0.85, 0.83, 1.2], "metallic": 0.1, "roughness": 0.5,
                                   "clearcoat": 0.6, "clearcoatRoughness": 0.2},
                   "Midnight Black": {"baseColor": [0.02, 0.02, 0.02, 1.2], "metallic": 0.1, "roughness": 0.4,
                                      "clearcoat": 0.8, "clearcoatRoughness": 0.1}},
        "defaultPaintName1": "Cardinal Red (faded)",
    })


def total_mass():
    return sum(d["w"] for d in nodes.values()) + 4 * (TIRE["numRays"] * 2 * (TIRE["nodeWeight"] + TIRE["hubNodeWeight"]))


if __name__ == "__main__":
    main_()
    print(f"nodes {len(nodes)}, beams {len(beams) + len(springs)}, triangles {len(triangles)}")
    print(f"sprung {m_sprung:.0f} kg, cg y={y_cg:.2f}, front corner {m_front / 2:.0f} kg, rear corner {m_rear / 2:.0f} kg")
    print(f"total approx {total_mass():.0f} kg")
    print("front spring", SPRING_F)
    print("rear spring ", SPRING_R)
