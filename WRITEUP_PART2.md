# Building Shoggoth Mini — Part 2: Shoggoth Mirror

<!-- SKELETON. Section headers and notes to write into; prose is yours.
     Notes in HTML comments are prompts and figures to hand, not copy.
     Numbers quoted below are measured and traceable to a commit or a tool run;
     re-check any of them before publishing, since the tuning is still moving. -->

---

## Introduction

<!-- Two sentences on what the mirror IS, before any implementation detail:
     it watches a face and mirrors the state back through movement.
     Part 1 opened with a GIF and that worked; lead with one here too.
     Link back to Part 1, which ended promising exactly this:
     "explore expressive motions (likely going beyond the 2D projection), add
     non-human sound, and potentially additional layers of perception." -->

Shoggoth-Mirror is a second-iteration on the original Shoggoth project, where now via the camera it perceives the user's face and mirrors the perceived state of the user back through movement of its tentacle

---

## Why a mirror

<!-- Picks up Part 1's "Why". The ELEGNT paper (arXiv 2501.12493) splits
     expressive movement into four: intention, attention, attitude, emotion.
     Worth saying which two you went after and why.

     The honest framing of ELEGNT: it is research-through-design. They hand-
     authored trajectories per scenario and ran a user study; nobody solves the
     max F + gamma*E optimisation online. So what carries over is the LENS, not
     an algorithm.

     The tentacle-with-no-face angle is the interesting constraint, not a
     limitation: if affect reads at all here it reads through movement alone. -->

Given that I'm working toward making Shoggoth a fully-interactive multi-modal 'device', my goal was to first have Shoggoth act as 'mirror' to the user. Perceiving the user's state of course is a fundamental part to human-robot interaction, so I figured having this first step would be 1) useful to debug how perception translates into conception of a user's 'state' and 2) a good springboard to explore a larger space of motions as well as including sound to develop better intuition how to make Shoggoth appear "lifelike", even as a mirror.

I looked to the ELEGNT paper as some inspiration for this phase. The authors split expressive momvement into four categories: intention, attention, attitude, and emotion. With this mirror phase, the emphasis is on attention and emotion. Attention of course in perceiving the user and deducing a state and emotion in trying to play back that that perceived state in an expressive enough motion that the user can intuit an 'emotional' state from Shoggoth. Given that Shoggoth's only form of expression is tentacle movement and sound, it's an interesting constraint. 


---

## System Design

<!-- The most distinctive decision in the project and worth its own section.

     States, transitions and timings are CSV tables the robot reads at runtime.
     tools/fsm_check.py reads the same tables and checks them. So does the
     diagram: --diagram emits mermaid from the parsed table, which cannot
     describe a machine other than the one that runs. A hand-drawn diagram is a
     second reading of the tables and can be wrong in ways nobody notices.

     Figure: verify/mirror_fsm.mmd (renders on GitHub in a ```mermaid fence)

     What the checker does, and why each pass exists:
       static        every symbol resolves
       determinism   for every (state, input), how many transitions fire?
       reachability  dead states, and states you can never leave
       replay        drive it from a real recording

     The clash map is the figure that earns its place: 62 clashing cells of 310
     collapsed to 0, and the picture showed they were nearly all ONE over-broad
     row rather than twenty separate problems. -->

(insert diagram)

The overall system flow for Shoggoth-Mirror can be best encapsulated at a high-level in the diagram above. Overall, the camera feed drives the perception. A variety of machine learning models under the MediaPipe framework from Google extract features like facial presence, pose, and blendshape (facial and eye movements). Those artifacts are fed into a variety of detection schema for inferring input states that go into the state machine. At the moment broken down into, the following inputs are: attention (is user facing Shoggoth), head gesture (did user nod 'yes' or shake 'no'), and Emotion (inferred from arousal and valence values).

These states are then fed into the state machine, which determines the overall state of the user that Shoggoth will mirror and then plays that state back via MotionWorker (which essentially carries the motion primitive and vocalization to be played based on the state). The state machine architecture is defined by state tables that I defined (4 tables overall, in shoggoth-mirror-state folder) which are parsed and then implemented in `affect/fsm.py`. See the transition definitions table below:

(insert table)


The parser converts the definitions laid out in the tables into executable transitions: each row's condition becomes a list of Terms, its hold time a float, and "Any except ALONE" an explicit list of source states. Nothing restates the table in code, so the checker, the generated diagram and Shoggoth are all reading the same 9 states and 16 rows.

Some insight behind some of the decisions:

1) For "Yes" and "No" detection, I set that as priority 1 as it should override any current state to communicate the fact that it acknowledges/mirrors the user's shaking/nodding.

2) Transitioning to ALONE when face is not present: this should override any previously registered emotion state

3) Transition to NOTICING: similar with 2), this should override any previously registered emotion state once attention is lost


---

## Perception: from a face to a number

<!-- MediaPipe FaceLandmarker (a bundle: BlazeFace + FaceMesh + a blendshape
     head), then the part that is actually yours: running two views through the
     project's own stereo calibration to get a head position in METRES.

     The measurement that validated the whole chain: median interpupillary
     distance 64.7 mm, on 95% of frames, against a human adult 55-72 mm. IPD is
     fixed on a real face, so every wobble in it is measurement error in the
     same units as the head position. Worth explaining why that is a better
     check than eyeballing the position trace.

     Arousal/valence: two independent sign tests, NOT a weighted sum. A single
     weighted combination projects the plane onto a line and collides the
     diagonally opposite quadrants -- angry (+arousal, -valence) and content
     (-arousal, +valence) both score zero. Good short explainer.

     Be honest that the blendshape weights are a hand-tuned guess, and that
     single-frame categorical emotion recognition is contested (Barrett et al.).
     What earns its keep is CHANGE over time.

     Figure: a plot_face_csv.py page from a real session. -->

---

## Motion: designing a vocabulary

<!-- side_side is the story. Anticipation and overshoot -- the classic animation
     trick -- and why the eye reads the CHANGE in speed more than the distance.

     Why a segment list beat a closed form: every number in the table is one a
     person chose (0.70 retreat, 1.20 overshoot, durations from the multipliers).
     A sum of sinusoids buries all three in phase relationships and you cannot
     tune one without disturbing the others.

     The servo ceiling as a DESIGN constraint, not a safety footnote: 7600
     ticks/s, measured in Part 1's characterisation. It sets how sharp a
     flourish can be, and the beat needs >= ~5 command points or the servo gets
     a shrug instead of a tick.

     The left/right discovery is a good short beat: the first version used motor
     2's axis (copied from slow_breathe) which is a NOD. 60 degrees is
     perpendicular -- motor 2 dead still at alignment 0.000, motors 1 and 3
     opposing at +/-0.866.

     Figures: exploration/out/primitives_mirror.png,
              exploration/out/side_side.png -->

---

## Making it react

<!-- The orchestrator: perception -> state machine -> motion, no model in it at
     all. Distinct from the author's `orchestrate`, which is an LLM in a voice
     loop calling primitives as tools.

     Two problems that only appear once it is live, both good writing:

     1. Primitives could not be interrupted. ALONE plays slow_breathe for 24.6 s
        and a state change had to wait it out -- someone walks up and stands
        there for most of half a minute. Cooperative cancellation with an eased
        exit brought it to ~0.5 s. Note WHY the ease matters: bailing mid-motion
        leaves the tentacle anywhere, and the reset would then snap it home.

     2. A nod was detected, entered and left without one command reaching a
        motor. The worker had ONE "latest wins" slot, and YES held exactly its
        600 ms min duration before NEUTRAL overwrote it. A state describes how
        things ARE; a gesture acknowledges something a person DID. Conflating
        them is what dropped it.

     The audio is a small nice beat: BodyAction carried an unused `sound` field
     from v1 for exactly this, so adding it threaded nothing new through. -->

---

## On verification

<!-- Part 1 did not need this section; this phase does. The argument:

     Several of these defects were only findable by tooling, and one was
     invisible until a checker ran against the real table.

     - affect_golden.py pins every quantity (labels, baselines, gesture spans,
       FSM traces) so a refactor can prove it changed nothing. It reported
       IDENTICAL across a library extraction that moved ~500 lines.
     - Its honest limit, worth stating: it pins BEHAVIOUR, not correctness. The
       'sad' bug below was real, was fixed, and the golden never moved -- because
       no fixture exercised it.
     - mirror_rehearsal.py runs the mirror's motion with tendons ATTACHED, going
       through the same MotionWorker the robot uses.

     The uncomfortable one to include: two bugs hid behind fixtures with no face
     in them, and a test that watched the command COUNT rise "verified" an
     interrupt that did not exist. Worth saying plainly -- it is the most useful
     paragraph in the section. -->

---

## Open Issues

<!-- Keep Part 1's spirit: name them, give the mechanism, do not soften.

     - Affect labels are twitchier than the machine's timings assume. 48% of
       runs were shorter than the 1 s exit threshold before the sign hysteresis;
       still worth re-measuring.
     - Arousal is the weak axis. The blendshape weights only ever ADD to a
       neutral face, so it rarely goes negative and saturates early. Hysteresis
       treats the symptom; the weights are the cause.
     - Transitions between motion families still cut. CONTENT -> EXCITED is two
       settings of one shape and could be continuous; CONTENT -> SAD cannot.
     - Head-pose signs never verified against the rig.
     - No fast abort. Ctrl+C is graceful, not an emergency stop.
     - grab and the arch sit closest to the encoder range; the margin shrinks on
       its own with every retension. -->
     - Big takeaway would be tune to optimnize hand-tuning of arousal/valence

---

## Lessons

<!-- Part 1's strongest section: concrete, surprising, with a mechanism.
     Five candidates, all real, roughly in order of how much they teach. -->

1. **A config loaded with no argument returns defaults, not your YAML.**
   <!-- tick_sign +1 against the YAML's -1 would have driven the tentacle
        backwards, with EVERY range guard passing, because the positions are
        perfectly legal -- just mirrored. Same trap had already cost a session
        via units_to_meters 0.05 vs 1.0, which made a 63 mm IPD read as 3.1 mm. -->

2. **The face detector died at 50 cm because of aspect ratio, not distance.**
   <!-- MediaPipe works on a ~192 px square. A 1920x760 eye scales by 0.1, so a
        150 mm face at 0.5 m (186 px in frame) arrives as 19 px -- right at
        BlazeFace's floor. Cropping to a square recovers it: 0.25 scale, 47 px. -->

3. **"Neutral" and "no face" were the same value.**
   <!-- An absence read as evidence of calm. Worse, NaN fails every comparison,
        so it escaped the neutral radius test and landed in the (False, False)
        quadrant: the robot would have mirrored SADNESS at someone whose
        expression it could not read. -->

4. **Hysteresis on the radius but not on the sign tests.**
   <!-- The deadband was a Schmitt trigger on distance from baseline, but once
        outside it the quadrant came from two bare comparisons. 17 of 28 label
        changes crossed a quadrant boundary rather than the deadband, twelve of
        them sad<->angry, with |d_arousal| under 0.02 on 28% of frames. -->

5. **A test can pass while measuring the wrong thing.**
   <!-- The interrupt "verified" by watching the command count rise -- which
        slow_breathe does whether or not it was cut short. The real test spies on
        which primitive started, and showed slow_breathe had never been
        interruptible at all. -->

---

## A note on AI-assisted development

<!-- Part 1's version was accurate and modest: Claude Code was fast at tooling
     and no help on hardware bugs that produce no error message.

     This phase is a different story and worth being straight about both halves.
     Much more of it was written that way -- the perception bench, the state
     machine checker, the motion primitives, the orchestrator. Several of the
     bugs in Lessons above were introduced in that code and then caught by other
     tooling written the same way.

     The line that seems true: it is good at building the thing that checks the
     thing, and the checks are what made the speed safe. It is still no help at
     all when the tentacle simply does not move and nothing anywhere errors. -->

---

## What's next

<!-- Where you actually are: making motion continuous across state transitions.
     CONTENT -> EXCITED first, since they are two settings of one shape, and the
     endpoint is a continuous motion layer where states set targets for
     amplitude, frequency and character rather than selecting from a menu.
     That is the architecture the affect/state split has been pointing at from
     the beginning. -->

---

## Credits & links

- Original project + writeup: Matthieu Le Cauchois — https://www.matthieulc.com/posts/shoggoth-mini/
- Part 1: [Getting It Working](README.md)
- ELEGNT paper: https://arxiv.org/pdf/2501.12493
- SpiRobs (tentacle) paper: https://arxiv.org/pdf/2303.09861
