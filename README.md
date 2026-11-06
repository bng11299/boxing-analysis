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
| Reported results | First baselines below. Video-disjoint split, **not** the paper's official one |

## First baselines

Six punch classes, 4,757 clips from the nine pre-segmented videos, 30 epochs,
seed 0, CPU. Held out V2/V5/V9 (976 clips, 20.5%).

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
- `split_by_video` holds out whole videos instead — weaker than
  subject-disjoint, but the strongest guarantee the release supports. The
  held-out set was fixed on label distribution alone, before any model was fit.
- The missing third channel is synthesised as an explicit **validity mask**, not
  a fabricated confidence. This matters: 46–70% of every clip is zero-padding,
  and the feature extractor reads that channel to drop it.
- V6 is **excluded by default** (`--include-v6` to opt in). Its segmentation
  would be ours, not the authors', and an off-by-one on the span would misalign
  every V6 clip while looking entirely plausible.

## Run it yourself

From a clean clone to a report for one clip. Nothing here needs an account, an
API key or a language model, and once step 2 is done nothing touches the network.
Windows PowerShell shown; Python 3.13 and `ffmpeg` on PATH.

**1. Install.**

```powershell
git clone <this repo>; cd boxing-comps
.\scripts\setup-env.ps1          # venv, CPU torch, the pose step; ends ENVIRONMENT OK
```

**2. Fetch the two pinned third-party checkpoints** (the only downloads; each is kept
only if its sha256 matches the pin):

```powershell
.\.venv\Scripts\python.exe scripts\fetch_weights.py
```

**3. Prove the chain works offline**, before any real data: a synthetic clip
goes through solo and two-boxer analysis with every network call refused. The
numbers it prints say nothing about boxing.

```powershell
.\.venv\Scripts\python.exe scripts\offline_smoke.py --allow-synthetic
.\.venv\Scripts\python.exe tests\test_offline.py      # the same, inside a fresh git clone
```

**4. Build the four model files from BoxingVI**, once. The corpus is on Google Drive
behind a sign-in, so you download it yourself (see *Getting the data*) into `data/`.
Each command below writes one file that `analyse_clip.py` reads by default:

| File | Made by | Time (CPU) |
|---|---|---|
| `results/punch_finder_v6.json` (punch-finder rule) | `python run_punch_finder.py --data data --video data/rgb/V6.mp4` | minutes |
| `results/sweep-2026-09-22/seed0.json` (reference reliability) | `python run_baseline.py --data data --stgcnpp both --pretrained weights/stgcnpp_ntu120_xsub_hrnet_j.pth --stgcnpp-epochs 20 --seed 0 --out results/sweep-2026-09-22/seed0.json` | ~35 min |
| `weights/classifier/stgcnpp-boxingvi-seed0.{pt,json}` (punch classifier) | `python train_classifier.py --data data --pretrained weights/stgcnpp_ntu120_xsub_hrnet_j.pth` (saves only if it reproduces the reference) | ~35 min |
| `results/punch_extent.json` (punch-length rule) | `python run_punch_extent.py --data data --classifier weights/classifier/stgcnpp-boxingvi-seed0.json` | minutes |

`run_punch_finder.py` needs V6's source video in `data/rgb/` (a YouTube link in the
release's `RGB_videos/` spreadsheet). All four files are gitignored: they come from
BoxingVI, whose licence grants use, not redistribution.

**5. Analyse a clip.** You give the stance; the tool never guesses one.

```powershell
python analyse_clip.py --video my_clip.mp4 --stance orthodox      # add --device cuda:0 on a GPU
```

Outputs land in `results/analyse/`: `X.pose.npz`, `X.punches.csv`,
`X.classified.csv` + `.json`, `X.report.xlsx`, `X.summary.txt`, `X.overlay.mp4`.

**Two boxers in one clip.** Track the people, say which track ids are which boxer
(read them off the tracked overlay), then analyse:

```powershell
python run_ingest.py --video bout.mp4 --track --weights weights/yolo11m-pose.pt --weights-sha256 29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2
python make_identity.py --tracks results/ingest/bout.tracks.npz --a 1 4 --b 2
python analyse_clip.py --two-boxer --video bout.mp4 --tracks results/ingest/bout.tracks.npz --identity results/ingest/bout.identity.csv --stance-a orthodox --stance-b southpaw
```

The identity you give is recorded as `by-hand, ungated`, and each boxer's report
says so in its first sentence.

**Optional: a reworded summary from Gemini.** The default summary is a fixed
template. `--narrate gemini` (solo clips only) sends the report's own sentences,
never video, frames or pose, to Gemini and keeps the reply only if every number
in it comes from those sentences. The key is read from the environment and
nowhere else. If there is no key, the call fails, or the reply fails the check,
you get the template and a one-line reason. Clips under a participant folder are
never sent.

```powershell
$env:GEMINI_API_KEY = "<your key>"
python analyse_clip.py --video my_clip.mp4 --stance orthodox --narrate gemini
```

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
