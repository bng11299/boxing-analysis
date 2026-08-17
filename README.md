# boxing-comps

Monocular boxing action recognition on the [BoxingVI](https://arxiv.org/abs/2511.16524)
benchmark. Senior comps project, Occidental College.

The benchmark ships 6,915 labelled punch clips with AlphaPose keypoints and
**reports no classification results**. Establishing the first ones is this repo's
initial deliverable, and it depends on nothing else in the project succeeding.

## Status

| | |
|---|---|
| Pipeline | Built and validated end-to-end on synthetic fixtures |
| Tests | 41 passing, weighted toward the loader and feature trust boundaries |
| Real corpus | **Not yet obtained** — see below |
| Reported results | None. Nothing here has been run on real data. |

## Getting the data

Only `Skeleton_data/` and `Annotation_files/` are needed. Skip `RGB_videos/` —
20 unedited source videos, and the keypoint baseline never touches pixels.

The corpus is on Google Drive and **requires sign-in**, so it cannot be fetched
programmatically: <https://drive.google.com/drive/folders/1Vyl8twJQ1qkqEPwhvfsrJsJ8nLQ92uoy>

Once downloaded, run the layout probe first — the on-disk arrangement inside
`Skeleton_data/` has never been inspected, and `data.py` reports what it finds
rather than assuming:

```bash
python run_baseline.py --data path/to/Dataset
```

That prints the layout and stops. Wire `load_corpus()` in `src/boxingvi/data.py`
against what it shows, then re-run.

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
