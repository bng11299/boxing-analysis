# How our numbers compare with published work

Working notes for the poster and paper (checked 2026-11-07). Our numbers come from
`docs/RESULTS.md` and are frozen at tag `results-freeze-2026-11-06`. Other works' numbers are
as their authors report them; we re-ran none of them.

## Why a headline number alone misleads

Three things decide whether two accuracies can be compared:

1. **Split.** If the same athlete appears in both training and test data, the score goes up,
   because the model can recognise the person instead of the punch. Our headline (row 1f)
   holds out whole videos and athletes (O55). Moving from the old split, where V2's instructor
   also appeared in training, to an athlete-disjoint one cost us 4 points of accuracy and
   8 points of macro-F1 (rows 1d to 1f).
2. **Classes and chance.** Four balanced classes give a 25% floor; our six imbalanced
   classes give a 31% floor by always guessing the commonest type (row 1h). Classes that
   differ only by hand (lead vs rear hook) are the hard part. Our weakest class is rear
   hook, with an F1 of 0.21 (row 1g).
3. **Input and setting.** Sensors and depth cameras see the motion directly. RGB models see
   gloves and context. Pose models see only 17 joints from one camera. Footage also differs:
   lab footage, shadowboxing, sparring, and broadcast bouts get harder in that order.

## Punch-type classification

| Work | Data / setting | Input, model | Classes (chance) | Split | Reported | Comparable? |
|---|---|---|---|---|---|---|
| **Ours (row 1f)** | BoxingVI, YouTube shadowboxing / bag / pads | 2D pose, ST-GCN++ pretrained, CPU | 6, imbalanced (0.31) | Held-out videos **and athletes** | **77.2% acc, 0.633 macro-F1** (3 seeds) | — |
| Ours, interpretable (row 1h) | same | 2D pose, hand features + random forest | 6 (0.31) | same | 67.4% acc, 0.494 macro-F1 | — |
| BoxingVI (Hegde et al., arXiv 2511.16524, 2025) | same dataset | — | 6 | Ships a 5,513 / 1,402 clip split | **No classification result** | Ours is the first baseline. Semantic Scholar lists 0 citing papers as of 2026-11-07 |
| BoxMAC (arXiv 2412.18204, 2024) | 15 pro boxers, real bouts | RGB; ViT best (CNN and ConvLSTM also tried) | 13 multi-label, punches **plus** defence/stance | Test clips from different videos (boxer overlap not stated) | 70.4–73.9% test acc | **Closest.** Similar band, harder label set, RGB. How multi-label accuracy is computed isn't stated |
| FACTS (arXiv 2412.16454, 2024) | Olympic Boxing (Stefański), league bouts | RGB, VideoMAE | 8 (hand × head / body / block / miss) | Stratified; by boxer or by video not stated | 83.25% acc | Partly. Different classes: target and outcome, not punch type. Head punches score F1 0.62–0.65, body punches 0.97–0.98 |
| Watanabe & Kameda (icSPORTS 2024) | **Computer-generated** 3D boxing simulation | OpenPose to ST-GCN++ | 4 (jab, straight, hook, uppercut) | Random over synthetic clips | 94% type, 88% hit/miss | **No.** Synthetic footage, random split, no lead/rear distinction. No result on real footage |
| Bhargav (blog post, not peer-reviewed) | Custom dataset | Pose, PoseConv3D | 6 | Not stated | 87.3% acc | **No.** No stated split or data |
| Manoharan et al. (PLOS One 2025, Hegde's group) | 8 elite boxers, lab | **Wrist IMUs** + video | 3 types per hand, after punch detection | Tested on unseen boxers | 89.9–97.5% per hand | Different sensor. Shows what's possible when the wrists are measured directly |
| Kasiri et al. (CVIU 2017) | Elite boxers, lab | **Overhead depth camera**, hierarchical SVM | Fine-grained punch types | Not checked (paywalled) | 97.3% acc | Different sensor and a fixed lab view |
| JAB AI (consumer app) | Users' phone videos | Not disclosed | Not disclosed | Not disclosed | ~70%, **developer-stated** in an App Store review reply | Developer's stated figure. Cite it as such |

## Finding punches in continuous footage

| Work | Task | Reported | Ours |
|---|---|---|---|
| Stefański, Kozak & Jach (*Entropy* 2024) | Is each frame a punch? Olympic bouts, one static camera | Punch-frame F1 0.49 (best method); 0.07 from raw images | Event F1 0.571 on V6 (row 3). Recall 0.585 and precision ≥0.633 on V5+V9 (row 5a) |
| Manoharan et al. (2025) | Punch vs no-punch from wrist IMUs | ~91–92% acc | — |
| Jabbr / DeepStrike (commercial) | Punch counts in pro fights | No published accuracy; compared fight by fight with CompuBox | — |

Frame-level and event-level F1 are different metrics, so this is a rough comparison, not a ranking.

## What we can claim

- **First result on BoxingVI** (77.2% / 0.633), on an athlete-disjoint split the benchmark
  itself doesn't enforce.
- **In the same band as the closest published RGB work** (BoxMAC 70–74%), using only 2D pose on
  a CPU. The class sets differ, so state it as "comparable range", not "beats".
- **Above JAB AI's developer-stated ~70%.**
- **The higher published numbers come with easier conditions.** The 87–97% results use
  synthetic footage, sensors, depth cameras, fewer classes, or splits that aren't stated.
  The main pattern is that numbers fall as conditions get closer to real sparring with
  unseen athletes.
- **Our weak spot matches theirs.** We struggle to tell lead from rear hooks, and
  Manoharan's group reports hooks confused with jabs. That supports the
  interpretability-gap analysis.

## Before citing in the paper

- Read BoxMAC's full text for how multi-label accuracy is computed and whether test boxers
  appear in training.
- Get Kasiri et al. 2017's class count and split; only an excerpt was read here.
- FACTS and BoxMAC are arXiv preprints. Check whether either has since been peer-reviewed.
