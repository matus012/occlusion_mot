# occlusion-mot in plain English

*One page. No jargon where I can avoid it. Last updated 2026-08-19.*

---

## The problem

A multi-object tracker follows people through a video and gives each person an ID. When
somebody walks behind a pillar, a van, or another person, the tracker loses them. When
they come out the other side, it usually calls them a **new person** with a **new ID**.

That is the failure this project attacks. It matters because almost every downstream use —
counting people, measuring flow, following one individual — breaks when identities keep
resetting.

**The baseline number to remember: ByteTrack keeps the right ID through an occlusion only
26% of the time.** Three out of four people come out the far side as a stranger.

## What I built

A **hidden-state module** that sits on top of the tracker. When a person disappears, it
does not delete them. It keeps a prediction of where they are while they are invisible,
and when a new detection appears it decides whether that is the same person coming back.

It uses two signals: **geometry** (where should they be by now, given how they were
moving?) and **appearance** (does this new box look like the person we lost?).

## What actually worked

Measured on MOT17, a standard public benchmark, on a half of the data held back from all
tuning.

| | keeps the ID through occlusion |
|---|---|
| ByteTrack baseline | 26% |
| + geometry | 31% |
| + appearance (off-the-shelf features) | 33% |
| + appearance (features I trained) | **34.5%** |

It also never made ordinary tracking worse — standard quality scores went slightly **up**,
not down, which is the thing that usually gets sacrificed for a gain like this.

## What did NOT work — and this is a real result, not a failure

The obvious next move was: train the appearance model on far more people, on a
supercomputer, and the gap should widen.

**I ran that experiment properly and it did not work.** 23 training runs on TUKE's PERUN
cluster (H200 GPUs), three random seeds each, with the success criteria written down and
locked *before* the runs so I could not move the goalposts afterwards.

The result: training on 12x more identities made the appearance model **much better at
recognising people** (a retrieval score climbed steadily, 0.41 → 0.61 and still rising)
but made the **tracker barely any better** (0.52 → 0.58, and the run-to-run random
variation alone was 0.078 — larger than the entire improvement).

So the honest finding is: **a better appearance model was not what stood between us and
the goal.** I reported that as a negative result rather than quietly dropping it.

## What the real bottleneck turned out to be

If the appearance model was not the limit, what was? I tested it directly by feeding the
tracker **perfect detections** — the ground-truth boxes, i.e. a hypothetical flawless
detector — and changing nothing else.

| | how many occlusions are even recoverable | keeps the ID |
|---|---|---|
| current detector | 57% | 34.5% |
| perfect detector | **97%** | **79%** |

That is decisive. **The detector is the bottleneck, not the appearance model.** The
tracker was never the thing holding this back; it was starved of detections. A better
detector has a lot of room to run.

Two honest caveats I keep attached to that number:

- 79% is a **ceiling, not a promise**. A real detector lands somewhere between 34.5% and
  79%. Measuring where is the experiment currently running.
- Even with perfect detections, ~19% of recoverable cases are still lost *inside* the
  tracker. Better detection cannot fix those.

## What is running right now

12 detector-training runs on PERUN, testing how much of that gap a *realistic* detector
closes. The detectors train only on datasets that share no images with the test set, so
the answer will be honest rather than flattering — an earlier version of this experiment
trained and tested on the same half, which made it look far better than it was.

## What I would say about this project in an interview

The engineering is real, but the part I would actually defend is the **discipline**:
success criteria written and frozen before runs, a negative result reported instead of
buried, and a caveat published that invalidates one of my own headline metrics
(a retrieval score turned out not to be comparable across experiment arms, so I demoted
it rather than quoting it). The bottleneck finding came out of taking the null result
seriously instead of tuning until something looked good.

## Where things stand

- **Works and holds up on held-out data:** the occlusion module improves identity
  retention and costs nothing in ordinary tracking quality.
- **Closed, negative:** scaling the appearance model does not get us to the target.
- **Open, promising:** the detector has large measured headroom; the run to quantify it
  is in flight.
- **Known limit:** a residual failure inside the tracker that no detector can fix.
