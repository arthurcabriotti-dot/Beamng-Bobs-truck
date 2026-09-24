"""Rough offline sanity check of the generated jbeam (NOT BeamNG's solver).

Loads the generated jbeam files, simulates the node/beam network with gravity and
simplified tyres (a vertical spring under each hub), and checks:
  * the structure settles without exploding
  * the ride height stays close to the modelled height (spring precompression is right)
  * suspension moves freely and the steering hydro turns both front wheels the correct way
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_truck import VEH, TIRE_R  # noqa: E402

DT = 1 / 2000.0


def load(parts):
    main = json.load(open(os.path.join(VEH, "bobs_truck.jbeam")))["bobs_truck"]
    acc = json.load(open(os.path.join(VEH, "bobs_truck_accessories.jbeam")))
    sections = [main] + [acc[p] for p in parts]
    ids, pos, mass = [], [], []
    beams = []  # a, b, k, c, pre
    hydros = []
    for sec in sections:
        w = 10.0
        for row in sec["nodes"][1:]:
            if isinstance(row, dict):
                w = row.get("nodeWeight", w)
                continue
            ids.append(row[0]); pos.append(row[1:4]); mass.append(w)
    idx = {n: i for i, n in enumerate(ids)}
    for sec in sections:
        st = {"beamSpring": 0, "beamDamp": 0, "beamPrecompression": 1}
        for row in sec["beams"][1:]:
            if isinstance(row, dict):
                st.update(row)
                continue
            beams.append((idx[row[0]], idx[row[1]], st["beamSpring"], st["beamDamp"], st["beamPrecompression"]))
        hst = {}
        for row in sec.get("hydros", [])[1:]:
            if isinstance(row, dict):
                hst.update(row)
                continue
            o = dict(hst)
            if len(row) > 2 and isinstance(row[2], dict):
                o.update(row[2])
            hydros.append((idx[row[0]], idx[row[1]], o["beamSpring"], o["beamDamp"], o.get("factor", 0)))
    return ids, idx, np.array(pos, float), np.array(mass, float), beams, hydros, main


def simulate(parts=(), steps=8000, steer=0.0, push=None, report=True):
    ids, idx, P, M, beams, hydros, main = load(parts)
    # tyre+rim mass lumped on hub nodes
    wheels = [r for r in main["pressureWheels"][1:] if isinstance(r, list)]
    tire_nodes = []
    for r in wheels:
        for n in (r[3], r[4]):
            M[idx[n]] += 22.0
            tire_nodes.append(idx[n])
    tire_nodes = np.array(tire_nodes)
    A = np.array([b[0] for b in beams] + [h[0] for h in hydros])
    B = np.array([b[1] for b in beams] + [h[1] for h in hydros])
    K = np.array([b[2] for b in beams] + [h[2] for h in hydros], float)
    C = np.array([b[3] for b in beams] + [h[3] for h in hydros], float)
    L_init = np.linalg.norm(P[A] - P[B], axis=1)
    pre = np.array([b[4] for b in beams] + [1.0] * len(hydros))
    hyd_factor = np.array([0.0] * len(beams) + [h[4] for h in hydros])
    V = np.zeros_like(P)
    X = P.copy()
    g = np.array([0, 0, -9.81])
    k_tire, c_tire = 320000.0 / 2, 3000.0 / 2
    maxv_hist = []
    for step in range(steps):
        inp = steer * min(1.0, max(0.0, (step - 3000) / 1500)) if steer else 0.0
        L0 = L_init * pre * (1 + hyd_factor * inp)
        d = X[B] - X[A]
        L = np.linalg.norm(d, axis=1)
        u = d / L[:, None]
        vrel = np.sum((V[B] - V[A]) * u, axis=1)
        f = K * (L - L0) + C * vrel
        F = np.zeros_like(X)
        np.add.at(F, A, u * f[:, None])
        np.add.at(F, B, -u * f[:, None])
        F += M[:, None] * g
        # tyres: hub node below radius -> vertical spring; strong horizontal damping (grip)
        pen = TIRE_R - X[tire_nodes, 2]
        on = pen > 0
        F[tire_nodes[on], 2] += k_tire * pen[on] - c_tire * V[tire_nodes[on], 2]
        F[tire_nodes[on], 0] += -c_tire * 4 * V[tire_nodes[on], 0]
        F[tire_nodes[on], 1] += -c_tire * 4 * V[tire_nodes[on], 1]
        # other nodes: ground contact
        low = X[:, 2] < 0
        F[low, 2] += 2e5 * (-X[low, 2]) - 500 * V[low, 2]
        if push is not None and step > 3000:
            F[idx[push[0]]] += np.array(push[1])
        V += F / M[:, None] * DT
        X += V * DT
        if step % 200 == 0:
            maxv_hist.append(float(np.abs(V).max()))
        if not np.isfinite(X).all():
            print("EXPLODED at step", step)
            return None
    return ids, idx, P, X, V, maxv_hist


def heading(X, idx, a, b):
    d = X[idx[b]] - X[idx[a]]
    return math.degrees(math.atan2(d[1], abs(d[0]))) * (1 if d[0] > 0 else -1)


if __name__ == "__main__":
    ok = True
    for parts in ((), ("bt_toolbox_uws", "bt_plowmount_western", "bt_plow_western")):
        r = simulate(parts)
        if r is None:
            sys.exit(1)
        ids, idx, P, X, V, mv = r
        dz = {n: X[idx[n], 2] - P[idx[n], 2] for n in ("fr2lb", "fr4lb", "fr7lb", "bc1lt", "bb3lt")}
        print(f"parts={parts or 'base'}  final max |v|={mv[-1]:.4f} m/s")
        for n, v in dz.items():
            print(f"   {n:6s} height change {v * 100:+.1f} cm")
        if mv[-1] > 0.05 or max(abs(v) for v in dz.values()) > 0.08:
            ok = False
    # steering: +1 input should be a right turn -> both front wheels' axes rotate clockwise from above
    r = simulate((), steps=6000, steer=1.0)
    ids, idx, P, X, V, mv = r
    hl, hr = heading(X, idx, "fw1l", "fw2l"), heading(X, idx, "fw1r", "fw2r")
    print(f"steer +1: left wheel axis yaw {hl:+.1f} deg, right wheel {hr:+.1f} deg (expect both < 0 = right turn)")
    ok &= hl < -20 and hr < -20 and abs(hl - hr) < 15
    print("OK" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)
