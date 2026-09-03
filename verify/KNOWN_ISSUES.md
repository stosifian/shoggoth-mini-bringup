# Known issues in the affect stack

Found while extracting `shoggoth_mini/affect/` and deliberately **left in
place** so the refactor's golden comparison stayed clean. A refactor that also
fixes things cannot prove it changed nothing. Each of these needs its own
commit, with its own golden delta looked at on purpose.

Reproduce anything here with `tools/affect_golden.py`.

---

## 1. ~~A face with unusable affect is labelled `sad` on the live path~~ FIXED

Fixed 2026-09-03. The finiteness guard now lives inside `quadrant()`, so it
applies to every driver and no future one can omit it. Both paths return
`unknown`. Golden is unchanged, because neither test take contains a frame with
a face and non-finite affect -- the bug was real but unexercised by the fixtures,
which is exactly the kind a golden cannot catch on its own.

Original report follows.

**Severity: real.** Silent, and it invents an emotion out of missing data.

`AffectState.update()` branches only on `have_face`. `classify_affect()` also
tests `np.isfinite` on arousal and valence. So a frame where the detector found
a face but the blendshape values are NaN takes different paths:

```
live  -> 'sad'
batch -> 'unknown'
```

`sad` because NaN fails every comparison: `hypot(nan, nan) < bar` is False, so
it leaves the neutral branch, then `da > a_thresh` and `dv > v_thresh` are both
False, and `(False, False)` is the sad quadrant. The robot would mirror sadness
at someone whose expression it could not read.

**Fix:** move the finiteness test into `AffectState.update()` alongside the
`have_face` check, and return `unknown`. Better still, put the guard inside
`quadrant()` so no future caller can reintroduce it — the pure function is the
right place for a precondition that belongs to the rule rather than to either
driver.

**Why not now:** it changes streaming labels on any take containing such a
frame, which is a golden delta. Worth doing next, and worth checking how often
it actually happens in a real recording first.

---

## 2. Gesture thresholds are tunable in the plot and fixed in the checker

**Severity: moderate.** Two tools disagreeing about the same take, which is the
class of bug the library was built to eliminate.

```
plot_face_csv.py:119  detect_head_gestures(t, yaw, pitch,
                          win=args.gesture_window, min_amp=args.gesture_amp)
fsm_check.py:409      detect_head_gestures(t, yaw, pitch)      # defaults
```

Tune `--gesture-amp` while looking at a plot, and `fsm_check` still uses 8.0 —
so the nod you just tuned into visibility never reaches the state machine, and
the two views of one recording disagree with no indication why.

**Fix:** give `fsm_check` the same flags and pass them through, or move the
values into `AffectConfig` and have both read the config. The second is better:
it is the same argument as for the thresholds, and it extends to the
orchestrator, which will need the identical numbers.

---

## 3. `_noise_sigma` is dead code

**Severity: cosmetic.** Superseded by `_rolling_sigma` when gesture detection
moved to a local noise floor. Still defined and exported. Harmless, but it is
the version with the bug that reported 31 shakes in a take containing one, so
leaving it callable is an invitation.

**Fix:** delete it, or keep it and say in the docstring that it is retained only
for comparison.

---

## Not bugs, but worth knowing

**Live and offline labels agree on 75–89% of frames, not 100%.** This is
structural, not a defect: a running median can see only the past, a whole-take
median sees everything. `plot_face_csv` shows both, shading by the live labels
and drawing the recomputed ones as a shadow ribbon. Treat divergence as a signal
about `BASELINE_WINDOW_S`.

**Head-pose signs are unverified.** `head_angles` yaw/pitch signs depend on
MediaPipe's convention and the rig mounting, and have never been checked against
a real face. Turn right, expect positive yaw; nod down, expect positive pitch.
`--flip-yaw` / `--flip-pitch` if not.

**The blendshape weights are a guess.** A full smile reads about +0.97, so the
gain saturates early. Tunable at `affect/config.py:BLENDSHAPE_WEIGHTS`.
