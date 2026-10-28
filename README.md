# boxing-comps

Monocular boxing action recognition on the [BoxingVI](https://arxiv.org/abs/2511.16524)
benchmark. Senior comps project, Occidental College.

The benchmark ships 6,915 labelled punch clips with AlphaPose keypoints and
**reports no classification results**. Establishing the first ones is this repo's
initial deliverable, and it depends on nothing else in the project succeeding.

## Status

| | |
|---|---|
| Pipeline | Built, validated, and **run on the real corpus** (2026-09-02) |
| Tests | 76 passing, weighted toward the loader and feature trust boundaries |
| Real corpus | Obtained 2026-09-02. **It is not what the paper describes** — see below |
| Reported results | Current: athlete-disjoint split V2/V3/V5/V9 ([docs/RESULTS.md](docs/RESULTS.md), O55). **Not** the paper's official split |

## First baselines

Six punch classes, 4,757 clips from the nine pre-segmented videos, 30 epochs,
seed 0, CPU. Held out V2/V5/V9 (976 clips, 20.5%).

> **Superseded 2026-10-27 (O55).** V2 and V3 show the same instructor, so this
> split is not athlete-disjoint. Current numbers are on the athlete-disjoint
> hold-out V2/V3/V5/V9 in [docs/RESULTS.md](docs/RESULTS.md). The table below is
> kept as the record of the first run.

| Arm | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Floor | majority-class | 0.205 | 0.057 |
| Floor | frequency-weighted random | 0.196 | 0.155 |
| Floor | static-pose (one observed frame) | 0.491 | 0.215 |
| Interpretable | **random forest** | **0.653** | **0.510** |
| Interpretable | SVM-RBF | 0.645 | 0.499 |
| Learned | ST-GCN | 0.552 | 0.407 |

**The cost of interpretability is negative here: −0.103 macro F1.** The random
forest over named temporal features beats the learned graph model, so on this
corpus interpretability is not paid for in accuracy. Its most important features
are rear-hand reach, elbow range and mean speed — which is what a coach would
say separates these punches.

### Is that comparison fair? (2026-09-07)

The obvious objection is that the learned arm was handicapped, so it was given
every advantage that could be given and re-run:

| | ST-GCN macro F1 | gap vs random forest |
|---|---:|---:|
| as first reported | 0.407 | -0.103 |
| + padding-masked pooling | 0.404 | -0.106 |
| + inverse-frequency class weighting (both arms) | 0.404 | -0.110 |
| + 60 epochs instead of 30 | 0.417 | -0.093 |

**The ranking does not move.** Three objections are now closed:

- *"The pooling averaged over padding."* It did -- `mean(dim=(2,3))` divided by
  25 frames when only ~10 are real -- and the fix is in. Padding is now provably
  unreachable: perturbing the padded frames moves the output logits by
  0.00000000, down from 0.142. A genuine bug, and not what was holding the
  learned arm back.
- *"You didn't weight the classes."* Now available via `--balanced`, applied to
  the forest, the SVM and both neural arms alike. Weighting only the arm that is
  losing would be a differently unfair comparison, not a fairer one.
- *"It was undertrained."* The opposite. ST-GCN reaches **0.921 train accuracy
  against 0.558 validation** -- a 0.363 generalisation gap; at 60 epochs it is
  0.979 / 0.569 and the gap *widens* to 0.410 for +0.011 validation accuracy.
  It memorises the training set. More epochs is not the missing ingredient;
  more clips is, which is what the learning curve independently shows.

One thing does change under weighting: **SVM-RBF becomes the strongest arm on
macro F1 (0.530), above the random forest's 0.514.** Which model leads depends
on whether classes are weighted, so both are reported rather than whichever
flatters.

Two cautions that belong next to any use of these numbers:

- **ST-GCN clears the static-pose floor by only 0.061 accuracy.** A single
  observed frame gets most of the way. The temporal model's use of time is
  therefore modest, and the macro-F1 gap (0.407 vs 0.215) says most of what it
  adds is on the rarer classes rather than overall.
- **`rear_hook` scores F1 = 0.000** for the random forest — not one correct
  prediction. It is the rarest class (7.2% of labels, 50 clips in validation).
  The 0.510 macro F1 is an average over a class the model never gets right.
  Class weighting does **not** rescue it: the balanced forest predicts
  `rear_hook` 14 times and gets 0 right, so this is not simply a rarity
  artefact. Its errors land on cross (21/50) and rear uppercut (20/50) -- the
  right hand, the wrong trajectory.

## What the release actually contains

The corpus does not match its own documentation, in five ways. Measured
2026-09-02 by downloading and auditing it:

| The paper and README say | The release ships |
|---|---|
| 20 source videos | **10** (V1–V10) |
| 18 athletes; subjects S1–S20 | **no subject identifier anywhere** |
| Official split S1–S15 / S16–S20 | **not reproducible** — nothing to split on |
| 6,915 clips | **4,766** pre-segmented (5,451 labelled rows counting V6) |
| COCO x, y, confidence | **two channels** — no confidence value ships |
| padded to 25 frames | true for 9 of 10; **V6 is raw per-frame pose** |

The paper is also internally inconsistent before any of this: it claims 18
athletes while numbering subjects S1–S20.

Consequences baked into the code rather than left to memory:

- `split_by_subject` **refuses** real clips and says why. The official split
  cannot be applied, and the guard exists so nobody quietly substitutes a
  clip-level split, which leaks athletes across folds.
- `split_by_video` holds out whole videos instead, and refuses any hold-out
  that puts one person on both sides (`data.PERSON_OF_VIDEO`: V1/V6 and V2/V3
  are one instructor each, O55). The original V2/V5/V9 set was fixed on label
  distribution alone, before any model was fit; V3 joined it on 2026-10-27
  because it shows V2's instructor.
- The missing third channel is synthesised as an explicit **validity mask**, not
  a fabricated confidence. This matters: 46–70% of every clip is zero-padding,
  and the feature extractor reads that channel to drop it.
- V6 is **excluded by default** (`--include-v6` to opt in). Its segmentation
  would be ours, not the authors', and an off-by-one on the span would misalign
  every V6 clip while looking entirely plausible.

## Getting the data

Only `Skeleton_data/` and `Annotation_files/` are needed (~50 MB). Skip
`RGB_videos/` — it holds no video at all, just a 4 KB spreadsheet of YouTube
links, and the keypoint baseline never touches pixels.

The corpus is on Google Drive and **requires sign-in**, so it cannot be fetched
programmatically: <https://drive.google.com/drive/folders/1Vyl8twJQ1qkqEPwhvfsrJsJ8nLQ92uoy>

`load_corpus()` is wired against the layout as it actually shipped, so a run is
just:

```bash
python run_baseline.py --data path/to/Dataset
```

The layout probe still runs first and prints what it found; `--layout-only`
stops there. If the probe ever reports something different from the table
below, trust the probe and not this README.

## Validating without the corpus

```bash
python -m pip install -r requirements.txt
python tests/test_pipeline.py
python run_baseline.py --synthetic
```

The synthetic run exercises every stage. Its accuracy figures are meaningless as
statements about boxing — the fixtures are trivially separable by construction —
and `run_baseline.py` will not write them to `results/` without
`--allow-synthetic`. What the run demonstrates is that the floors, both model
arms and the metrics are wired correctly, which is visible in the static-pose
baseline landing between chance and ceiling rather than at either.

## What it runs

| Arm | Model | Purpose |
|---|---|---|
| Floor | majority-class, frequency-weighted random | Chance |
| Floor | **static-pose** — one frame per clip | How much of the task posture alone solves. A temporal model that fails to beat this has not shown it uses time. |
| Interpretable | Random Forest, SVM-RBF over named temporal features | Feature importances; the arm whose errors can be attributed |
| Learned | compact ST-GCN, ~200 lines of plain PyTorch | What an off-the-shelf learned model achieves |

The difference between the last two is reported as the **cost of
interpretability** — the accuracy given up by choosing a model whose mistakes can
be explained.

## Why not pyskl / mmcv

They need compilation, are pinned to older Python, and do not install on Windows
with Python 3.13. The value of a first baseline is that someone else can check it,
so the model is written from scratch in plain PyTorch instead. It is small enough
to train on CPU: no GPU is assumed anywhere in this repo.

## Licence and citation

BoxingVI is used for academic work with citation, granted by Ravi Hegde
(2026-08-17). Baseline results are to be contributed back to the authors' repo —
a commitment, not a courtesy.

```bibtex
@article{kumar2025boxingvi,
  title={BoxingVI: A Multi-Modal Benchmark for Boxing Action Recognition and Localization},
  author={Kumar, Rahul and Baghel, Vipul and Singh, Sudhanshu and Badatya, Bikash Kumar
          and Yadav, Shivam and Srinivasan, Babji and Hegde, Ravi},
  journal={arXiv preprint arXiv:2511.16524},
  year={2025}
}
```
