"""Capture every quantity the affect stack computes, as exact values.

Run BEFORE the refactor to record a golden, and AFTER to compare. Anything that
differs is a behaviour change, whether or not it was intended -- the point of a
refactor is that this file's output is byte-identical.

  python golden.py before.json
  python golden.py after.json
  python golden.py --diff before.json after.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "exploration"))
sys.path.insert(0, str(REPO))

TAKES = ["exploration/out/face_state_test.csv",
         "exploration/out/face_state_test_v2.csv"]


def digest(a) -> str:
    """Order-sensitive hash of an array, so a diff is one line not thousands."""
    b = np.asarray(a)
    if b.dtype == object or b.dtype.kind in "US":
        raw = "\x1f".join(map(str, b.ravel())).encode()
    else:
        raw = np.ascontiguousarray(b, dtype=np.float64).tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def capture():
    import plot_face_csv as pfc
    import face_probe as fp
    import fsm_check as fc

    out = {}
    for take in TAKES:
        p = REPO / take
        d = pfc.load(p)
        t, yaw, pitch = d["t"], d["yaw"], d["pitch"]
        face = np.isfinite(yaw)
        rec = {}

        # --- batch classification -----------------------------------------
        labels, (a_base, v_base) = pfc.classify_affect(
            d["arousal"], d["valence"], have_face=face)
        rec["batch_labels_digest"] = digest(labels)
        rec["batch_labels_head"] = list(labels[:25])
        rec["batch_counts"] = {s: int((labels == s).sum())
                               for s in sorted(set(labels))}
        rec["baseline_arousal"] = float(a_base)
        rec["baseline_valence"] = float(v_base)

        # --- streaming classification (the live path) ----------------------
        st = fp.AffectState()
        stream = [st.update(float(t[i]), float(d["arousal"][i]),
                            float(d["valence"][i]), bool(face[i]))
                  for i in range(len(t))]
        rec["stream_labels_digest"] = digest(np.array(stream, object))
        rec["stream_counts"] = {s: int(sum(1 for x in stream if x == s))
                                for s in sorted(set(stream))}
        rec["stream_final_baseline"] = [float(x) for x in st.baseline]
        rec["stream_vs_batch_agreement"] = float(
            np.mean(np.array(stream, object) == labels))

        # --- gestures -------------------------------------------------------
        nod, shake = pfc.detect_head_gestures(t, yaw, pitch)
        rec["nod_digest"], rec["shake_digest"] = digest(nod), digest(shake)
        rec["nod_spans"] = [[round(a, 3), round(b, 3)] for a, b in pfc._spans(t, nod)]
        rec["shake_spans"] = [[round(a, 3), round(b, 3)] for a, b in pfc._spans(t, shake)]
        rec["rolling_sigma_yaw_digest"] = digest(pfc._rolling_sigma(t, yaw))
        rec["rolling_sigma_pitch_digest"] = digest(pfc._rolling_sigma(t, pitch))

        # --- attention ------------------------------------------------------
        att = [bool(abs(yaw[i]) < fp.ATTEND_YAW_DEG
                    and abs(pitch[i]) < fp.ATTEND_PITCH_DEG)
               if face[i] else False for i in range(len(t))]
        rec["attending_digest"] = digest(np.array(att, float))
        rec["attending_frames"] = int(sum(att))

        # --- affect from blendshapes (scalar grid, no take needed) ----------
        rec["affect_from_blend"] = [
            list(fp.affect_from_blend(b)) for b in (
                {}, {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0},
                {"mouthFrownLeft": 0.7, "browDownLeft": 0.4},
                {"eyeWideLeft": 0.9, "jawOpen": 0.6, "browInnerUp": 0.5})]

        # --- head pose geometry ---------------------------------------------
        def rot_y(dg):
            c, s = np.cos(np.radians(dg)), np.sin(np.radians(dg))
            M = np.eye(4); M[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]; return M
        rec["head_angles"] = [[round(float(x), 6) for x in fp.head_angles(rot_y(a))[:3]]
                              for a in (0, 15, -30, 55)]

        # --- the state machine ----------------------------------------------
        tables = REPO / "shoggoth-mirror-state"
        inputs, aliases, states, srows, timing, trans = fc.load_tables(tables)
        vectors = list(fc.legal_vectors())
        res = fc.replay(p, states, trans, "ALONE", timing)
        rec["fsm_trace_digest"] = digest(np.array(res["trace"], object))
        rec["fsm_transitions"] = [[round(x[0], 3), x[1], x[2], x[3]]
                                  for x in res["log"]]
        rec["fsm_occupancy"] = {k: round(v, 4) for k, v in res["occupancy"].items()}
        rec["fsm_blocked"] = len(res["blocked"])
        rec["fsm_lost"] = len(res["lost"])
        out[take] = rec

    # table-level facts, take-independent
    tables = REPO / "shoggoth-mirror-state"
    inputs, aliases, states, srows, timing, trans = fc.load_tables(tables)
    vectors = list(fc.legal_vectors())
    grid = []
    for s in states:
        for v in vectors:
            firing = [x for x in trans
                      if s in x.sources and x.dst != s and fc.satisfied(x, v)]
            top = min((x.priority if x.priority is not None else 99)
                      for x in firing) if firing else -1
            tied = [x for x in firing
                    if (x.priority if x.priority is not None else 99) == top]
            grid.append(len(tied))
    out["_tables"] = {"states": states, "n_vectors": len(vectors),
                      "clash_cells": int(sum(1 for c in grid if c > 1)),
                      "grid_digest": digest(np.array(grid, float))}
    return out


def main():
    if sys.argv[1] == "--diff":
        a = json.loads(Path(sys.argv[2]).read_text())
        b = json.loads(Path(sys.argv[3]).read_text())
        bad = 0
        for take in sorted(set(a) | set(b)):
            for k in sorted(set(a.get(take, {})) | set(b.get(take, {}))):
                va, vb = a.get(take, {}).get(k), b.get(take, {}).get(k)
                if va != vb:
                    bad += 1
                    print(f"  DIFF {take} :: {k}\n    before {va}\n    after  {vb}")
        print(f"\n{'IDENTICAL' if not bad else str(bad) + ' DIFFERENCE(S)'}")
        return 1 if bad else 0
    Path(sys.argv[1]).write_text(json.dumps(capture(), indent=1, sort_keys=True))
    print(f"wrote {sys.argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
