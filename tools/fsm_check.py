"""Check the Shoggoth-Mirror state machine tables before anything is built from them.

The tables are three CSVs kept in a spreadsheet (States / Timing / Transitions,
plus Inputs). This reads them as the source of truth and answers the questions
that are tedious and unreliable to answer by eye:

  STATIC       does every symbol resolve? Conditions that name an input nobody
               declared, transitions to states that do not exist, states with no
               timing row -- these are the errors that look fine in a spreadsheet
               and fail the moment something tries to execute them.

  DETERMINISM  for every (state, input) pair, how many transitions fire? Two at
               the same priority is a coin flip: the machine's behaviour depends
               on row order in a CSV. This is the check that cannot be done by
               inspection, because the clashes live in combinations nobody
               thought to picture.

  REACHABILITY which states can actually be entered from the start state, and
               from which states can you never return? An unreachable state is
               dead code; a state that cannot reach ALONE is a trap the robot
               never comes home from.

  REPLAY       drive the machine from a real face_probe recording and report
               which states are actually occupied and how often it switches.
               The three checks above find logical errors; this one finds the
               tuning errors -- thresholds that never satisfy on real noisy
               signals, or fire so often the robot twitches.

Run:
  python tools/fsm_check.py shoggoth-mirror-state/
  python tools/fsm_check.py shoggoth-mirror-state/ --replay out/face.csv

On the timing abstraction: for the exhaustive pass a condition "X == 1 for more
than 1.5 s" is treated as satisfiable whenever X == 1, i.e. every timer that
COULD be elapsed is assumed elapsed. That is deliberately the worst case. A race
that only appears once two timers both mature is still a race, and asking
whether it is reachable in wall-clock time is a separate (and much harder)
question than whether the table defines what to do about it.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))     # the package

# The machine itself lives in the library so the checker and the orchestrator
# run the SAME state machine rather than two that are meant to agree. Everything
# below is checking and drawing -- no rules.
from shoggoth_mini.affect.fsm import (BOOLS, EMOTIONS, MIN_DURATION,  # noqa: E402
                                      PREVIOUS, Machine, Term, Transition,
                                      active_emotion, audit_condition,
                                      legal_vectors, load_tables, min_dur,
                                      parse_condition, run_vectors, satisfied,
                                      tidy, to_mermaid)

# =============================================================================
# input space
# =============================================================================
# =============================================================================
# checks
# =============================================================================
class Report:
    def __init__(self):
        self.errors, self.warnings = [], []

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def check_static(rep, inputs, aliases, states, states_rows, timing, transitions):
    print("\n=== STATIC ===")
    known = set(states)

    for t in transitions:
        for s in t.sources:
            if s not in known:
                rep.err(f"{t.label()}: source state '{s}' is not declared")
        if t.dst != PREVIOUS and t.dst not in known:
            rep.err(f"{t.label()}: target state '{t.dst}' is not declared")
        if t.special == "UNPARSED":
            rep.err(f"{t.label()}: condition not understood: {t.cond_raw!r}")
        if not t.terms and t.special is None:
            rep.err(f"{t.label()}: empty condition")
        for term in t.terms:
            if term.name not in aliases:
                rep.err(f"{t.label()}: '{term.name}' is not a declared input "
                        f"(declared: {', '.join(sorted(aliases))})")
        if t.priority is None:
            rep.warn(f"{t.label()}: no priority set")
        if re.search(r"\bor\b", t.cond_raw, re.I):
            rep.err(f"{t.label()}: the grammar has no OR, so "
                    f"'{t.cond_raw}' would silently parse as something "
                    f"narrower. Split it into two rows.")
        leftover = audit_condition(t.cond_raw, t.terms)
        if leftover and t.special is None:
            rep.warn(f"{t.label()}: parser IGNORED {leftover} in "
                     f"'{t.cond_raw}'")
        if t.dst in t.sources and len(t.sources) > 1:
            rep.warn(f"{t.label()}: '{t.dst}' is in its own source set "
                     f"(self-transition)")

    for s in states:
        if s not in timing:
            rep.err(f"state '{s}' has no row in the Timing table")
    for s in timing:
        if s not in known:
            rep.err(f"Timing has a row for '{s}', which is not a declared state")

    for r in states_rows:
        if r.get("State") and not r.get("Body"):
            rep.warn(f"state '{r['State']}' has no Body defined")

    used = {term.name for t in transitions for term in t.terms}
    for i in inputs:
        bare = i.split(":", 1)[-1].strip()
        if i not in used and bare not in used:
            rep.warn(f"input '{i}' is declared but never used in a condition")

    print(f"  {len(states)} states, {len(transitions)} transition rows, "
          f"{len(inputs)} inputs")
    print(f"  expanded to {sum(len(t.sources) for t in transitions)} concrete "
          f"(source -> target) edges")


def check_determinism(rep, states, transitions, vectors):
    print("\n=== DETERMINISM ===")
    clashes: dict[tuple, list] = {}
    fires_from = {s: 0 for s in states}

    for s in states:
        for v in vectors:
            firing = [t for t in transitions
                      if s in t.sources and t.dst != s and satisfied(t, v)]
            if firing:
                fires_from[s] += 1
            if len(firing) < 2:
                continue
            top = min((t.priority if t.priority is not None else 99)
                      for t in firing)
            tied = [t for t in firing
                    if (t.priority if t.priority is not None else 99) == top]
            if len(tied) > 1:
                key = (s, tuple(sorted(t.row for t in tied)))
                clashes.setdefault(key, []).append(v)

    if not clashes:
        print("  no nondeterminism: every (state, input) resolves to one winner")
    for (s, rows), vecs in sorted(clashes.items()):
        rows_desc = []
        for r in rows:
            t = next(x for x in transitions if x.row == r)
            rows_desc.append(f"row {r} -> {t.dst} [p{t.priority}]  {t.cond_raw}")
        rep.err(f"in {s}: {len(rows)} transitions tie at the top priority "
                f"for {len(vecs)} input combination(s)\n      "
                + "\n      ".join(rows_desc)
                + f"\n      e.g. {compact(vecs[0])}")

    dead = [s for s, n in fires_from.items() if n == 0]
    for s in dead:
        rep.err(f"state '{s}' has no outgoing transition under ANY legal input "
                f"-- it is a trap")


def compact(v: dict) -> str:
    emo = active_emotion(v) or "none"
    return (f"Face={v['Face']} Attention={v['Attention']} "
            f"Yes={v['Yes']} No={v['No']} Emotion={emo}")


def check_reachability(rep, states, transitions, vectors, start):
    print("\n=== REACHABILITY ===")
    edges = {s: set() for s in states}
    for s in states:
        for t in transitions:
            if s not in t.sources:
                continue
            if t.dst == PREVIOUS:
                edges[s].update(states)      # can return anywhere it came from
                continue
            if t.dst != s and any(satisfied(t, v) for v in vectors):
                edges[s].add(t.dst)

    seen, frontier = {start}, [start]
    while frontier:
        cur = frontier.pop()
        for nxt in edges[cur]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    for s in states:
        if s not in seen:
            rep.err(f"state '{s}' is unreachable from {start}")
    print(f"  reachable from {start}: {len(seen)}/{len(states)}")

    # can every state get home?
    home = "ALONE" if "ALONE" in states else start
    rev = {s: set() for s in states}
    for s, outs in edges.items():
        for o in outs:
            rev[o].add(s)
    seen2, frontier = {home}, [home]
    while frontier:
        cur = frontier.pop()
        for prv in rev[cur]:
            if prv not in seen2:
                seen2.add(prv)
                frontier.append(prv)
    for s in states:
        if s not in seen2:
            rep.err(f"state '{s}' can never reach '{home}' -- trap")
    print(f"  can reach {home}: {len(seen2)}/{len(states)}")

    for s in states:
        outs = sorted(edges[s])
        print(f"    {s:9} -> {', '.join(outs) if outs else '(nowhere)'}")


# =============================================================================
# replay against a real recording
# =============================================================================
def face_csv_to_vectors(path: Path):
    """Decode a face_probe recording into the machine's input vectors.

    This is the tool's job, not the library's: the machine should not know what
    a CSV is. The orchestrator will build the same vectors from live perception,
    which is what makes the two comparable.
    """
    import numpy as np
    from plot_face_csv import load as load_face
    from shoggoth_mini.affect import detect_head_gestures

    d = load_face(path)
    t = d["t"]
    nod, shake = detect_head_gestures(t, d["yaw"], d["pitch"])

    emo_col = None
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "state" in rows[0]:
        emo_col = [r["state"] for r in rows]

    vecs = []
    for i in range(len(t)):
        face = int(np.isfinite(d["yaw"][i]))
        att = int(d["attending"][i] == 1) if np.isfinite(d["attending"][i]) else 0
        emo = None
        if emo_col:
            raw = emo_col[i].strip().capitalize()
            emo = raw if raw in EMOTIONS else None    # unknown / calibrating
        elif face:
            emo = "Neutral"                # take predates affect labelling
        if not face:
            emo = None                     # nothing seen, nothing asserted
        v = {"Face": face, "Attention": att,
             "Yes": int(nod[i]), "No": int(shake[i])}
        for e in EMOTIONS:
            v[e] = int(e == emo)
        vecs.append(v)
    return t, vecs, (emo_col is not None)


def replay(path: Path, states, transitions, start, timing):
    print(f"\n=== REPLAY: {path.name} ===")
    t, vecs, had_states = face_csv_to_vectors(path)
    if not had_states:
        print("  ! this CSV has no 'state' column (recorded before affect "
              "labelling); frames with a face are treated as Neutral")

    # Input coverage FIRST. A replay that never leaves the start state is
    # ambiguous -- it can mean the machine is wrong, or that the recording never
    # contained the inputs needed to leave.
    print("  input coverage:")
    for k in BOOLS + EMOTIONS:
        n = sum(v[k] for v in vecs)
        note = "   <- never true" if n == 0 else ""
        print(f"    {k:10} {100*n/len(vecs):5.1f}% of frames{note}")
    if not any(v["Face"] for v in vecs):
        print("  ! no frame in this recording contains a face, so the machine "
              "cannot leave ALONE. This take cannot validate the table.")

    res = run_vectors(t, vecs, states, transitions, timing, start)
    total = sum(res["occupancy"].values()) or 1.0
    log = res["log"]
    print(f"  {len(t)} frames, {t[-1]-t[0]:.1f}s, "
          f"{len(log)} transitions ({len(log)/max(t[-1]-t[0],1e-9):.2f}/s)")
    print("  occupancy:")
    for s_ in states:
        pct = 100 * res["occupancy"][s_] / total
        bar = "#" * int(pct / 2)
        flag = "" if res["occupancy"][s_] > 0 else "   <- never entered"
        print(f"    {s_:9} {pct:5.1f}%  {bar}{flag}")
    if log:
        print("  first transitions:")
        for ts, a, b, row in log[:12]:
            print(f"    {ts:7.2f}s  {a:9} -> {b:9}  (row {row})")
    if res["blocked"]:
        print(f"  {len(res['blocked'])} frame(s) where a transition was "
              f"satisfied but held back by min duration")
    if res["lost"]:
        print(f"  {len(res['lost'])} frame(s) where more than one transition "
              f"fired and priority picked the winner")
    return res


EMO_COLOR = {"Neutral": "0.62", "Content": "tab:green", "Excited": "gold",
             "Sad": "tab:blue", "Angry": "tab:red"}


def plot_clashes(states, transitions, vectors, out: Path | None, title: str):
    """States x input-combinations, coloured by how many transitions fire.

    The determinism check prints one paragraph per clash, which for a table with
    a single over-broad row means twenty paragraphs that all say the same thing.
    As a grid the shape is obvious instead: a clash caused by one bad row draws
    a stripe across every state it applies to, and a clash caused by genuinely
    conflicting local rules draws isolated cells.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap, BoundaryNorm

    grid = np.zeros((len(states), len(vectors)), int)
    culprits: dict[int, int] = {}
    for si, s in enumerate(states):
        for vi, v in enumerate(vectors):
            firing = [t for t in transitions
                      if s in t.sources and t.dst != s and satisfied(t, v)]
            if not firing:
                grid[si, vi] = 0                      # stay put
                continue
            top = min((t.priority if t.priority is not None else 99)
                      for t in firing)
            tied = [t for t in firing
                    if (t.priority if t.priority is not None else 99) == top]
            grid[si, vi] = 1 if len(tied) == 1 else 2
            if len(tied) > 1:
                for t in tied:
                    culprits[t.row] = culprits.get(t.row, 0) + 1

    cmap = ListedColormap(["#f2f2f2", "#8fbf8f", "#d24b4b"])
    fig, ax = plt.subplots(figsize=(max(9, len(vectors) * 0.34),
                                    2.0 + len(states) * 0.42))
    ax.imshow(grid, aspect="auto", cmap=cmap,
              norm=BoundaryNorm([0, 0.5, 1.5, 2.5], cmap.N), interpolation="nearest")

    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states)
    ax.set_xticks(range(len(vectors)))
    ax.set_xticklabels([short(v) for v in vectors], rotation=90, fontsize=6.5,
                       family="monospace")
    ax.set_xlabel("input combination  (F=Face A=Attention Y=Yes N=No, then emotion)")
    n_clash = int((grid == 2).sum())
    ax.set_title(f"{title}\n{n_clash} clashing cells of "
                 f"{grid.size} (state x input)", fontsize=11)

    for x in range(len(vectors) + 1):
        ax.axvline(x - 0.5, color="white", lw=0.5)
    for y in range(len(states) + 1):
        ax.axhline(y - 0.5, color="white", lw=0.8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               ["#f2f2f2", "#8fbf8f", "#d24b4b"]]
    ax.legend(handles, ["stay (no transition)", "one winner", "TIE - undefined"],
              loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False,
              fontsize=9)

    if culprits:
        ranked = sorted(culprits.items(), key=lambda kv: -kv[1])
        lines = []
        for row, n in ranked[:6]:
            t = next(x for x in transitions if x.row == row)
            lines.append(f"row {row} -> {t.dst}: {n} cells")
        ax.text(1.005, 0.0, "rows involved in ties\n" + "\n".join(lines),
                transform=ax.transAxes, va="bottom", fontsize=8.5, color="0.2",
                family="monospace")

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    if out:
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"  -> {out}")
    else:
        plt.show()

    if culprits:
        print("\n  rows involved in ties, most first:")
        for row, n in sorted(culprits.items(), key=lambda kv: -kv[1]):
            t = next(x for x in transitions if x.row == row)
            print(f"    row {row:3d} -> {t.dst:14} {n:4d} cells   {t.cond_raw}")


def short(v: dict) -> str:
    emo = active_emotion(v) or "none"
    return (f"{'F' if v['Face'] else '-'}{'A' if v['Attention'] else '-'}"
            f"{'Y' if v['Yes'] else '-'}{'N' if v['No'] else '-'} {emo[:4]}")


def plot_replay(res, states, transitions, out: Path | None, title: str):
    """Logic-analyser view: inputs on top, resulting state underneath.

    The two panels share a time axis on purpose. Almost every question about a
    state machine is "why did it do that THEN", and the answer is always an
    input a moment earlier -- so the eye needs to travel vertically, not between
    two separate figures.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.asarray(res["t"], float)
    vecs = res["vecs"]
    fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                           gridspec_kw={"height_ratios": [1.15, 1]})
    fig.suptitle(title, fontsize=12)

    # --- inputs -----------------------------------------------------------
    a = ax[0]
    rows = list(reversed(BOOLS))
    for k, name in enumerate(rows):
        vals = np.array([v[name] for v in vecs], float)
        a.fill_between(t, k + 0.12, k + 0.12 + 0.62 * vals, step="post",
                       color="tab:blue", alpha=.75, lw=0)
        a.axhline(k + 0.12, color="0.85", lw=.8)
    # emotion is one-hot, so it draws as a single categorical lane rather than
    # five binary rows that can never overlap
    ey = len(rows)
    cur, start_i = None, 0
    emo_seq = [active_emotion(v) for v in vecs]
    for i, e in enumerate(emo_seq + [None]):
        if e != cur:
            if cur is not None and cur in EMO_COLOR:
                a.axvspan(t[start_i], t[min(i, len(t) - 1)],
                          ymin=(ey + 0.12) / (ey + 1.0), ymax=(ey + 0.74) / (ey + 1.0),
                          color=EMO_COLOR[cur], alpha=.55, lw=0)
            cur, start_i = e, i
    a.set_ylim(0, ey + 1.0)
    a.set_yticks([k + 0.4 for k in range(len(rows))] + [ey + 0.43])
    a.set_yticklabels(rows + ["Emotion"])
    a.set_ylabel("inputs")
    a.grid(alpha=.25, axis="x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=EMO_COLOR[e], alpha=.6)
               for e in EMOTIONS]
    a.legend(handles, EMOTIONS, loc="upper left", bbox_to_anchor=(1.005, 1.0),
             frameon=False, fontsize=9, title="Emotion")

    # --- state ------------------------------------------------------------
    b = ax[1]
    idx = {s: i for i, s in enumerate(reversed(states))}
    trace = res["trace"]
    run_start, cur = 0, trace[0]
    for i, s in enumerate(trace + [None]):
        if s != cur:
            y = idx[cur]
            b.barh(y, t[min(i, len(t) - 1)] - t[run_start], left=t[run_start],
                   height=0.62, color="tab:cyan", alpha=.75, lw=0)
            cur, run_start = s, i
    for ts, src, dst, row in res["log"]:
        for axis in ax:
            axis.axvline(ts, color="k", lw=.7, ls="--", alpha=.45)
        b.annotate(f"r{row}", (ts, idx[dst] + 0.42), fontsize=7.5,
                   ha="center", color="0.25")
    # near-misses: satisfied but held back, or beaten on priority. These are the
    # events that explain an ABSENCE of a transition, which is otherwise the
    # hardest thing to see in a trace.
    if res["blocked"]:
        bt = [x[0] for x in res["blocked"]]
        b.plot(bt, [idx[x[1]] - 0.42 for x in res["blocked"]], "v",
               ms=4, color="tab:orange", alpha=.8, label="held by min duration")
    if res["lost"]:
        lt = [x[0] for x in res["lost"]]
        b.plot(lt, [idx[x[1]] - 0.42 for x in res["lost"]], "x",
               ms=5, color="tab:red", alpha=.8, label="priority broke a tie")
    b.set_yticks(range(len(states)))
    b.set_yticklabels(list(reversed(states)))
    b.set_ylim(-0.8, len(states) - 0.2)
    b.set_ylabel("state")
    b.set_xlabel("time (s)")
    b.grid(alpha=.25, axis="x")
    if res["blocked"] or res["lost"]:
        b.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False,
                 fontsize=9)

    fig.tight_layout(rect=[0, 0, 0.88, 0.97])
    if out:
        fig.savefig(out, dpi=130)
        print(f"  -> {out}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", type=Path, help="directory holding the CSVs")
    ap.add_argument("--start", default="ALONE", help="start state")
    ap.add_argument("--replay", type=Path, default=None,
                    help="a face_probe CSV to drive the machine with")
    ap.add_argument("--plot", nargs="?", const="SHOW", default=None,
                    metavar="PNG",
                    help="draw the replay as a timing diagram; give a path to "
                         "save, or pass bare to open a window")
    ap.add_argument("--diagram", type=Path, default=None, metavar="MMD",
                    help="write a mermaid state diagram generated from the "
                         "parsed table; with --replay, edges carry fire counts")
    ap.add_argument("--clash-map", nargs="?", const="SHOW", default=None,
                    metavar="PNG",
                    help="draw the state x input clash grid from the exhaustive "
                         "pass; give a path to save, or pass bare for a window")
    args = ap.parse_args()

    inputs, aliases, states, states_rows, timing, transitions = load_tables(args.tables)
    vectors = list(legal_vectors())
    rep = Report()

    print(f"tables: {args.tables}")
    print(f"input space: {len(vectors)} legal combinations "
          f"(of {2**len(BOOLS) * len(EMOTIONS)} raw)")

    check_static(rep, inputs, aliases, states, states_rows, timing, transitions)
    check_determinism(rep, states, transitions, vectors)
    if args.start not in states:
        rep.err(f"start state '{args.start}' is not declared")
    else:
        check_reachability(rep, states, transitions, vectors, args.start)

    if args.diagram:
        fired = None
        if args.replay:
            r = replay(args.replay, states, transitions, args.start, timing)
            fired = {t.row: 0 for t in transitions}
            for _, _, _, row in r["log"]:
                fired[row] = fired.get(row, 0) + 1
        args.diagram.write_text(
            to_mermaid(states, states_rows, transitions, timing, fired))
        print(f"\n=== DIAGRAM ===\n  -> {args.diagram}")

    if args.clash_map:
        out = None if args.clash_map == "SHOW" else Path(args.clash_map)
        plot_clashes(states, transitions, vectors, out, args.tables.name)

    if args.replay:
        res = replay(args.replay, states, transitions, args.start, timing)
        if args.plot:
            out = None if args.plot == "SHOW" else Path(args.plot)
            plot_replay(res, states, transitions, out,
                        f"{args.replay.name} — {args.tables.name}")

    print("\n=== SUMMARY ===")
    for e in rep.errors:
        print(f"  ERROR   {e}")
    for w in rep.warnings:
        print(f"  warn    {w}")
    print(f"\n  {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
