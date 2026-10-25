# Commands: Classifier, punch extent, naming punches, end-to-end eval

Moved verbatim from the project rules' `## Commands` on 2026-10-22. Add new recipes here, not to the project rules.

Classifier for the pipeline: `python train_classifier.py --data data --pretrained weights/stgcnpp_ntu120_xsub_hrnet_j.pth` (~35 min CPU; saves `weights/classifier/stgcnpp-boxingvi-seed0.pt` + `.json` manifest only if it reproduces results/sweep-2026-09-22/seed0.json (0.8145 / 0.7207, 20 epochs)) ·

Punch-extent rule (once, after the classifier): `python run_punch_extent.py --data data --classifier weights/classifier/stgcnpp-boxingvi-seed0.json` -> `results/punch_extent.json` (ratios from the training clips; tested on V6) ·

Name the punches: `python classify_punches.py --pose results/ingest/<clip>.pose.npz --punches results/punches/<clip>.punches.csv --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --stance orthodox [--extent results/punch_extent.json] [--labels ...]` (needs `results/punch_extent.json`; every punch in a clip is read over the clip's median punch length, ADR-0014; writes `<clip>.classified.csv` + `.json` provenance) ·

End to end on V5 + V9 (finder + classifier vs every annotated span, blinded label audit): `python eval_pipeline_v5v9.py --data data --poses results/ingest --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --stance orthodox [--videos-dir data/rgb] [--judgments ...]` · E2E tests: `python tests/test_e2e.py` ·

Window cost on V6: `python eval_classify_v6.py --data data --classifier weights/classifier/stgcnpp-boxingvi-seed0.json` · Classify tests: `python tests/test_classify.py`
