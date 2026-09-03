"""The mirror state machine: parse the tables, evaluate them, step the machine.

The tables are the source of truth and they live in a spreadsheet, so this is a
parser over free text as much as it is an interpreter. That is a real weakness
and audit_condition exists because of it -- see its docstring.

This module is the MACHINE, not a tool for inspecting one. It knows nothing
about CSV recordings, matplotlib or the checker's report format: callers hand it
input vectors and it says what the machine does with them. That split is what
lets the offline checker and the live orchestrator run the same state machine
rather than two that are meant to agree.
"""

from __future__ import annotations

import csv
import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path


ANY = "__ANY__"
PREVIOUS = "Previous state"
MIN_DURATION = "__MIN_DURATION__"

# Curly quotes and non-breaking spaces come free with spreadsheet exports.
_TIDY = {"“": '"', "”": '"', "‘": "'", "’": "'",
         " ": " ", "–": "-", "—": "-"}


def tidy(s: str) -> str:
    for a, b in _TIDY.items():
        s = s.replace(a, b)
    return s.strip()


@dataclass
class Term:
    """One `Input == value` clause, optionally with a hold duration."""
    name: str
    value: int
    hold_s: float | None = None

    def __str__(self):
        d = f" for >{self.hold_s}s" if self.hold_s is not None else ""
        return f"{self.name}=={self.value}{d}"


@dataclass
class Transition:
    row: int
    src_raw: str
    sources: list[str]
    dst: str
    cond_raw: str
    terms: list[Term]
    special: str | None          # MIN_DURATION for the YES/NO return rows
    priority: int | None
    problems: list[str] = field(default_factory=list)

    def label(self) -> str:
        return f"row {self.row}: {self.src_raw} -> {self.dst}"


_DUR = re.compile(r">\s*([\d.]+)\s*(seconds?|secs?|s|milliseconds?|ms)\b", re.I)
_TERM = re.compile(r"\b([A-Za-z_][A-Za-z_ ]*?)\s*==\s*([01])\b")


def parse_duration(text: str) -> float | None:
    m = _DUR.search(text)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2).lower()
    return val / 1000.0 if unit.startswith("m") else val


def parse_condition(text: str) -> tuple[list[Term], str | None]:
    """-> (terms, special). Conjunctions only; `&` and `and` are equivalent."""
    text = tidy(text)
    if not text:
        return [], None
    if "==" not in text:
        # the YES/NO return rows: a trigger expressed in prose
        if "min duration" in text.lower():
            return [], MIN_DURATION
        return [], "UNPARSED"

    clauses = re.split(r"\s*&\s*|\s+and\s+", text, flags=re.I)
    terms: list[Term] = []
    for c in clauses:
        m = _TERM.search(c)
        if not m:
            continue
        terms.append(Term(m.group(1).strip(), int(m.group(2)), parse_duration(c)))
    return terms, None


# Words the condition grammar treats as prose. Whatever is left after the terms
# and durations are consumed, minus these, is text the parser IGNORED -- which
# is the failure mode that matters, because it is the silent one.
_NOISE = {"for", "longer", "than", "is", "are", "being", "returned", "detected",
          "seconds", "second", "secs", "sec", "s", "ms", "milliseconds", "and",
          "the", "a", "of", "state", "machine", "was", "in", "before", "after",
          "min", "duration", "return", "to", "landmarks", "on", "timeout",
          "i.e", "e.g", "no", "face"}


def audit_condition(text: str, terms: "list[Term]") -> list[str]:
    """Tokens in the raw condition the parser did not account for.

    The grammar is undocumented and the parser takes what it recognises, so the
    real risk is not a syntax error -- those surface -- but a clause quietly
    dropped. `A == 1 or B == 1` parses as `A == 1`: the split knows only `&` and
    `and`, and _TERM.search returns the FIRST match in a clause. A row like that
    would be checked, drawn and executed as something narrower than written,
    with nothing anywhere saying so.
    """
    t = tidy(text)
    for term in terms:
        t = re.sub(re.escape(term.name) + r"\s*==\s*[01]", " ", t)
    t = _DUR.sub(" ", t)
    t = re.sub(r"[(),.;:>&/\-]", " ", t)
    return [w for w in t.split()
            if w.lower() not in _NOISE and not w.replace(".", "").isdigit()]


def expand_sources(raw: str, states: list[str]) -> list[str]:
    """`Any except A, B` -> every state but A and B. `Any` -> all."""
    t = tidy(raw)
    low = t.lower()
    if not low.startswith("any"):
        return [t]
    if "except" not in low:
        return list(states)
    excl_raw = t[low.index("except") + len("except"):]
    excluded = {e.strip() for e in excl_raw.split(",") if e.strip()}
    return [s for s in states if s not in excluded]


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [{tidy(k): tidy(v or "") for k, v in row.items() if k}
                for row in csv.DictReader(fh)]


def load_tables(d: Path):
    def one(prefix):
        hits = sorted(d.glob(f"{prefix}*.csv"))
        if not hits:
            sys.exit(f"no {prefix}*.csv in {d}")
        return read_csv_rows(hits[0])

    inputs_rows = one("Inputs")
    states_rows = one("States")
    timing_rows = one("Timing")
    trans_rows = one("Transitions")

    # "Emotion: Content" is declared with a prefix but referenced bare.
    inputs, aliases = [], {}
    for r in inputs_rows:
        name = r.get("Inputs") or next(iter(r.values()))
        inputs.append(name)
        aliases[name] = name
        if ":" in name:
            aliases[name.split(":", 1)[1].strip()] = name
    states = [r["State"] for r in states_rows if r.get("State")]

    timing = {}
    for r in timing_rows:
        if r.get("State"):
            timing[r["State"]] = r.get("Min Duration", "")

    transitions = []
    for i, r in enumerate(trans_rows, start=2):     # row 1 is the header
        src_raw = r.get("From", "")
        dst = r.get("-> To") or r.get("To") or ""
        cond = r.get("Conditions", "")
        pr = r.get("Priority", "")
        terms, special = parse_condition(cond)
        transitions.append(Transition(
            row=i, src_raw=src_raw, sources=expand_sources(src_raw, states),
            dst=tidy(dst), cond_raw=cond, terms=terms, special=special,
            priority=int(pr) if pr.strip().isdigit() else None))

    return inputs, aliases, states, states_rows, timing, transitions


EMOTIONS = ["Neutral", "Content", "Excited", "Sad", "Angry"]
BOOLS = ["Face", "Attention", "Yes", "No"]


def active_emotion(v: dict) -> str | None:
    """Which emotion is asserted, or None when no face was seen."""
    return next((e for e in EMOTIONS if v[e]), None)


def legal_vectors():
    """Every input combination the world can actually present.

    Enumerating all 2^n would manufacture races out of states that cannot
    coexist, so the declared invariants are applied as filters:
      * exactly one Emotion at a time (declared in the Inputs table)
      * Yes and No are mutually exclusive (the detector scores the larger swing)
      * Attention implies Face -- attention is measured FROM a detected face
      * no Face implies NO emotion asserted -- all five at 0. Neutral is a
        claim about a face that was looked at; with no face there is nothing to
        claim. Reporting an absence as neutral is what let a rule saying
        "Neutral == 1" fire when the person LEFT.
    """
    for face, att, yes, no in itertools.product((0, 1), repeat=4):
        if att and not face:
            continue
        if yes and no:
            continue
        if (yes or no) and not face:
            continue          # a nod is detected from landmarks
        # with no face there is exactly one emotion vector: all zero
        for emo in (EMOTIONS if face else [None]):
            v = {"Face": face, "Attention": att, "Yes": yes, "No": no}
            for e in EMOTIONS:
                v[e] = int(e == emo)
            yield v


def satisfied(t: Transition, v: dict) -> bool:
    """Worst-case: any hold timer that could be mature is assumed mature."""
    if t.special == MIN_DURATION:
        return True                       # min duration always eventually elapses
    if t.special == "UNPARSED" or not t.terms:
        return False
    return all(v.get(term.name) == term.value for term in t.terms)


def min_dur(timing, state) -> float:
    raw = tidy(timing.get(state, "")).lower()
    if not raw or raw == "-":
        return 0.0
    m = re.match(r"([\d.]+)\s*(ms|s|seconds?)?", raw)
    if not m:
        return 0.0
    val = float(m.group(1))
    return val / 1000.0 if (m.group(2) or "").startswith("m") else val


def to_mermaid(states, states_rows, transitions, timing, fired=None) -> str:
    """Mermaid state diagram, emitted from the PARSED table.

    Generated from the same Transition objects the checker evaluates and the
    replay executes, so it cannot describe a machine other than the one that
    runs. A diagram drawn by hand from the CSV is a second reading of the tables
    and can be wrong in ways nobody notices; this one is wrong only if the
    parser is, and the parser is what everything else already trusts.

    Multi-source rows ("Any except ...") would otherwise draw one edge per state
    and bury the graph -- 50 edges from 22 rows. They come out of a single ANY
    node instead, which keeps the deliberate transitions readable while still
    showing the fallbacks exist.

    `fired` maps row -> count from a replay; when given, every edge carries how
    often it actually fired, so the unexercised parts of the table are visible.
    """
    body = {r["State"]: (r.get("Body") or "").strip() for r in states_rows}
    out = ["stateDiagram-v2", "    direction LR"]

    for st in states:
        b = body.get(st, "")
        b = b.replace("Play ", "").replace(" motion primitive", "")
        b = b.replace(" motion", "").replace('"', "").strip()
        md = min_dur(timing, st)
        extra = ""
        if md:
            extra = f" [min {md*1000:.0f}ms]" if md < 1 else f" [min {md:.1f}s]"
        if b or extra:
            out.append(f"    {st}: {b}{extra}")

    out.append(f"    [*] --> {states[0]}")
    has_any = has_prev = False
    for t in transitions:
        cond = tidy(t.cond_raw).replace(":", " ").replace('"', "'")
        cond = (cond[:52] + "...") if len(cond) > 55 else cond
        tag = f"r{t.row}" + (f" p{t.priority}" if t.priority is not None else "")
        if fired is not None:
            tag += f" x{fired.get(t.row, 0)}"
        dst = t.dst
        if dst == PREVIOUS:
            dst, has_prev = "PREV", True
        src = t.sources[0]
        if len(t.sources) > 1:
            src, has_any = "ANY", True
        out.append(f"    {src} --> {dst}: {tag} - {cond}")
    if has_any:
        out.append("    ANY: any state, see the row for its exclusions")
    if has_prev:
        out.append("    PREV: whichever state it came from")
    return "\n".join(out) + "\n"


# =============================================================================
# stepping
# =============================================================================
class Machine:
    """The running machine: current state, and what it does with an input vector.

    Deliberately ignorant of where inputs come from. The checker feeds it
    vectors decoded from a recording; the orchestrator will feed it vectors
    built from live perception. Neither can drift from the other, because
    neither owns the rules.

    Hold timers are tracked per INPUT, not per transition: "Face == 1 for > 1.5s"
    asks how long Face has been 1, which is a property of the signal and shared
    by every row that tests it.
    """

    def __init__(self, states, transitions, timing, start):
        self.states, self.transitions, self.timing = states, transitions, timing
        self.start = start
        self.cur, self.prev_state = start, None
        self.entered = None
        self.held: dict[str, float] = {}
        self._prev_v: dict | None = None

    def _ok(self, tr, t, v) -> bool:
        if tr.special == MIN_DURATION:
            return (t - self.entered) >= min_dur(self.timing, self.cur)
        if not tr.terms:
            return False
        for term in tr.terms:
            if v.get(term.name) != term.value:
                return False
            if term.hold_s and (t - self.held.get(term.name, t)) < term.hold_s:
                return False
        return True

    def step(self, t: float, v: dict):
        """Advance one frame. Returns (state, fired_row_or_None, status).

        status is 'fired', 'blocked' (a transition was satisfied but the state
        has not served its minimum) or 'idle'. 'blocked' is worth surfacing:
        a transition repeatedly held back is a min duration that is too long,
        and from the outside it looks exactly like a condition that never
        satisfies.
        """
        if self.entered is None:
            self.entered = t
        for k, val in v.items():
            if self._prev_v is None or self._prev_v.get(k) != val:
                self.held[k] = t
        self._prev_v = dict(v)

        firing = [tr for tr in self.transitions
                  if self.cur in tr.sources and tr.dst != self.cur
                  and self._ok(tr, t, v)]
        if firing and (t - self.entered) < min_dur(self.timing, self.cur):
            return self.cur, firing[0], "blocked"
        if not firing:
            return self.cur, None, "idle"

        firing.sort(key=lambda x: x.priority if x.priority is not None else 99)
        win = firing[0]
        dst = self.prev_state if win.dst == PREVIOUS else win.dst
        dst = dst or self.start
        self.prev_state, self.cur, self.entered = self.cur, dst, t
        return self.cur, win, "fired" if len(firing) == 1 else "fired_tie"


def run_vectors(t, vectors, states, transitions, timing, start):
    """Drive a machine through a whole sequence, collecting what it did."""
    m = Machine(states, transitions, timing, start)
    occupancy = {s: 0.0 for s in states}
    log, blocked, lost, trace = [], [], [], []
    for i in range(len(t)):
        before = m.cur
        cur, tr, status = m.step(float(t[i]), vectors[i])
        if status == "blocked":
            blocked.append((t[i], before, tr.dst, tr.row))
        elif status in ("fired", "fired_tie"):
            if status == "fired_tie":
                lost.append((t[i], before, [tr.row]))
            log.append((t[i], before, cur, tr.row))
        trace.append(cur)
        dt = (t[i + 1] - t[i]) if i + 1 < len(t) else 0.0
        occupancy[cur] += dt
    return dict(t=t, vecs=vectors, trace=trace, log=log, blocked=blocked,
                lost=lost, occupancy=occupancy)
