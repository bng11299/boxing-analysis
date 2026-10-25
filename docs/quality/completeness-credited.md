## `feat/completeness-credited` -- 2026-10-09

The completeness test on CREDITED fragments only, no linking (`completeness.find_bursts_credited`,
`olympic_completeness.py --credited [--min-punches N]`). Branch from `feat/boxer-linking` (1d67459).

- **Decision (ADR-0012 Option 2, Brendan, 2026-10-09):** two-boxer identity is parked until W9. Pose-only linking swaps boxers and
  absorbs spectators, and two corner-colour attempts failed their non-circular gate (torso colour does not separate the boxers in
  this footage). The completeness question for Phase 1 is answered by testing only the fragments that are credited with a labelled
  punch. No identity is needed: a ByteTrack id never changes person by design, so each credited fragment is one person for its whole
  length. An id switch inside a fragment is the tracker's residual error and is NOT detected here.
- **Design.** Population: every track id credited (`olympic.attribute`) with at least `min_punches` (default 1) labelled punches.
  Per fragment: its rows are split into segments as `boxer_segments` does (the id never changes, so only a hole longer than the bridge
  splits it); bridge, wrist signal and detection, the short-segment and bodyless rules, the exposure mask (processed segments, bridged
  frames included, at least one hand valid) and the assertion that every burst lies in its fragment's exposure are the SAME code as the
  linked boxers' (`completeness._scan_segments`, extracted from `find_bursts`; the 266 existing linking tests are unchanged and green).
  Ownership: a label is OWNED by the fragment it is credited to. Phase 1 matches each fragment's bursts to its own labels, phase 2
  matches the remaining bursts of all fragments to the UNATTRIBUTED labels (credited to no one), each label once; both phases use the
  existing `_match` (optimal within the phase, ties to the burst that sorts first by (frame, key), here the lower track id). Two
  co-present fragments are two people and are kept apart: a burst near a label owned by another fragment is not explained by it.
  Rates, the whole-video density and the seeded simple random sample are the existing ones; the CSV adds `track_id`, the fragment's
  box centre and height (pixels) at the burst frame (interpolated across a bridged hole) and `time_mss` (m:ss.s, rounded to a tenth
  first) so the person can be found by eye. Outputs `results/olympic_completeness_credited.json` and `..._credited_sample.csv`
  (the linking-based files are never overwritten). `--reacquire` with `--credited` is refused; so is `--min-punches` without
  `--credited`.
- **Decision beyond the brief:** a label credited to a track id that is NOT in the population (fewer than `min_punches` credited
  punches) belongs to a person this test does not follow. It is neither owned nor unattributed: it explains nothing and is counted as
  `credited_to_excluded`. Treating it as unattributed would let a person we do not follow excuse the bursts of those we do (mutation 8).
- **The stated limits (in the module docstring, the json `bias`, the CLI print and here; WORDING CORRECTED in Iteration 2, which
  replaced "can only UNDERCOUNT").** This sees only credited fragments. Missed punches outside them (boxer time in a fragment with no
  credited punch, punches too close to call) are never seen, so coverage can only lose; but fragment-level detections can DUPLICATE
  one physical punch when two overlapping ids follow the same person. So neither a bout-level upper nor a lower bound follows
  without de-duplication. A clean result is weak evidence of completeness; a dirty one shows labels are missing inside the credited
  fragments, though the count may include duplicates. A credited fragment can be a non-boxer wrongly credited (a referee, a
  coach, a spectator); the fragments with exactly one credited punch are the least certain and are counted.
- **Tests:** new `tests/test_completeness_credited.py`, 75 checks on planted scenes (two co-present credited fragments with a
  simultaneous burst, an uncredited referee, a one-punch fragment; `min_punches` 1/2/3 and bad values; an ambiguous label explains one
  burst once across fragments and the tie goes to the lower id; own labels first; a bridged-frame peak counted in numerator and
  exposure with its interpolated box; a 3-row segment adds no exposure and its label is outside it; a 2-frame hole stays one
  segment and a 3-frame hole ends it; the exposure RuntimeError; a punch credited to an id missing from the table refused (Iteration 2 widens this to any attribution that differs from the table's own); box_at / mss including
  59.96 s = 1:00.0; the CLI end to end on a planted dataset: exact `track_id`, box centre and height, `time_s`, `time_mss` for a burst
  on a bridged frame and one after the first minute, `inclusion_prob` = n/N = 0.666667, reproducible sample, `--credited
  --reacquire` refused with nothing written, `--min-punches 0 / -3` and `--min-punches` alone refused, default outputs leave sentinel
  linking files untouched, no credited punch runs to the end with null rates).

**Mutation check (change applied in a scratch copy; the named suite must fail)**
| # | Mutation | Suite | Result | First failing checks |
|---|---|---|---|---|
| 1 | ownership ignored (phase 1 skipped, every label unattributed) | credited | caught, 9 fail | P's own match; Q's burst not explained by P's label; S |
| 2 | uncredited fragments included in the population | credited | caught, 9 | population is {1, 2, 4}; the referee's burst not counted; min_punches=2 |
| 3 | one-to-one removed (every burst takes its nearest label) | credited / linking | caught, 2 / 12 | exactly one burst explained by the unattributed label; label used once / twelve existing matching tests |
| 4 | exposure from all rows instead of processed segments | credited | caught, 4 | bridged peak: 400 exposure frames; 3-row tail adds none; label outside exposure |
| 5 | burst box from the fragment's first row | credited | caught, 2 | exact CSV values; bridged-frame box |
| 6 | box not interpolated on a bridged frame | credited | caught, 3 | box_at; exact CSV values; bridged-frame box |
| 7 | m:ss.s truncates instead of rounding first | credited | caught, 1 | 59.96 s is 1:00.0 |
| 8 | label credited to an excluded fragment treated as unattributed | credited | caught, 2 | credited_to_excluded; still unexplained |
| 9 | burst-in-exposure assertion removed from the shared scan | credited / linking | caught, 1 / 1 | the RuntimeError test (both suites) |
| 10 | CLI: `--credited --reacquire` accepted | credited | caught, 1 | refused with exit 2 |
| 11 | CLI: credited default names are the linking-based ones | credited | caught, 2 | defaults leave the sentinels untouched; the linking run writes its own names |
| 12 | CLI: `inclusion_prob` = 1 instead of n/N | credited | caught, 1 | 0.666667 |
| 13 | CLI: csv writes the frame in the track_id column | credited | caught, 1 | exact values |
| 14 | `min_punches` ignored | credited | caught, 3 | min_punches=2 / =3 |

**Numbers (29 Olympic videos, rule `punch_finder_v6.json`, 0.1 s tolerance, `results/*.json`, gitignored; nothing viewed or judged)**
| Quantity | Credited, min 1 | Credited, min 2 | Linking (default, for comparison) |
|---|---|---|---|
| fragments in the population (with exactly 1 credited punch) | 1,245 (679) | 566 (0) | two identities per video |
| exposure minutes (summed over fragments) | 161.4 (91.1 with at least one fragment in exposure) | 96.1 | 146.4 |
| bursts / unexplained | 10,077 / 7,318 | 6,227 / 4,342 | 7,984 / 5,766 |
| unexplained per exposure minute | 45.4 | 45.2 | 39.4 |
| bursts per exposure minute | 62.5 | 64.8 | 54.5 |
| own labels in exposure per exposure minute | 14.9 (2,407 in exposure, 16 outside) | 18.1 (1,741, 3) | 10.8 (1,580) |
| labels owned / unattributed / credited to an excluded fragment | 2,423 / 2,182 / 0 | 1,744 / 2,182 / 679 | 1,585 / 3,020 / n.a. |
| labels matched (own + unattributed) | 1,473 + 1,286 = 2,759 | 1,053 + 832 = 1,885 | 973 + 1,245 = 2,218 |
| sample N (population), inclusion probability | 7,318, 0.00547 | 4,342, 0.00921 | 5,766, 0.00694 |
| segments / short / bodyless | 2,758 / 260 / 1 | 1,380 / 116 / 1 | |
| labels per whole-video boxer-minute (a different population) | 22.1 | 22.1 | 22.1 |

- The credited population holds more of the labels (2,423 owned of 4,605, against 1,585 on the two linked identities) because it needs
  no identity, and the unattributed share falls from 65.6% to 47.4% (the labels attribution itself could not call). The rates are NOT
  a finding about the labels: the populations differ (credited ids include referees, coaches and spectators; the exposure is summed
  over fragments, so two people on screen together count twice, and one person under two overlapping ids would too), the rule is V6's,
  and unexplained bursts are candidates until the sample is classified. Samples (40 each, seed 20261007) are for YOU to classify;
  nothing was viewed or judged.
- 3,850 of the 10,077 bursts (38%) lie in the 679 one-punch fragments, 2,904 of them unexplained. That is where a wrongly credited
  non-boxer would hide, which is why `--min-punches 2` was run beside the default.

**Tests (all suites):** `test_completeness_credited.py` 75 (new), `test_linking.py` 266 (unchanged), `test_tracks.py` 79,
`test_olympic.py` 33, `test_hands.py` 38, `test_labels.py` 50, `test_ingest.py` 124, `test_env.py` 36, `test_pipeline.py` 260.
**Security pass:** the only new inputs are two integers (`--min-punches`, validated whole and >= 1 at the CLI and in the function,
bool refused) and a track id taken from the attributions (refused when it is not in the table); no new path is built from user input,
no dependency, the CLI computes everything before writing, JSON written with `allow_nan=False`, a burst outside its exposure still
raises.

**Residual risks:** (1) the limits above: credited fragments only, so a clean result is weak evidence, and duplicate ids can inflate a dirty one; (2) a credited fragment can be a
non-boxer, and one-punch fragments carry 38% of the bursts; (3) an id switch inside a fragment is undetected; (4) the same person under
two overlapping ids is counted twice (Iteration 2 counts the likely cases, `possible_duplicates`, and does not subtract them), so the pooled exposure is fragment-minutes, not video time; (5) the rule is V6's (AlphaPose,
24 fps) and untuned for 50 fps sparring; (6) the 29 tables are bound by file name only; (7) matching is optimal within each phase,
ties go to the lower track id.

### Iteration 2 -- Astra 6 review of `feat/completeness-credited` (2026-10-09; verdict FIX FIRST: 1 blocking, 2 not)

| # | Finding | Fix | Evidence |
|---|---|---|---|
| B1 blocking | Attributions were not bound to the table that produced them: with the same id set but the people swapped (a re-tracked video), attributions of table A passed with table B went through every check and silently changed the population, matches and rates | `completeness.check_attrs_bound`, called by BOTH `find_bursts_credited` and `find_bursts`: recompute `olympic.attribute(events, table)` (default parameters) and require, event by event, the same credited track id, wrist, distance, runner-up distance, ambiguity flag and hand-clear flag (`ValueError` naming the event). What is checked is exactly that, not "ids exist in the table": the earlier missing-id test is kept and checked first. The CLI passes the attributions `load_video` made (computed once per video there) and the test recomputes them once more; nothing else changed | same ids, ownership swapped (ids 1 / 2 credited 1,1,1,2,2,2 in table A and 2,2,2,1,1,1 in table B): refused by both functions, accepted with the table's own attributions; credit removed, ambiguity flipped and wrist changed after the fact are refused. Attributions made with non-default `attribute` parameters would be refused too by construction (not separately tested; pass the defaults) |
| N1 wording | "can only UNDERCOUNT" overstated the inference: two overlapping ids on one person each report the same unlabelled punch | Wording replaced in the module docstring, `BIAS_CREDITED` (json `bias`, CLI print), the CLI docstring, the project rules and above: restricting to credited fragments loses missed punches outside them (coverage can only lose), while fragment detections can duplicate a punch, so neither a bout-level upper nor a lower bound follows without de-duplication. New number `possible_duplicates` (per fragment, per video, pooled; `possible_duplicate_pairs` beside it): an unexplained burst with an unexplained burst of ANOTHER credited fragment within the matching tolerance (0.1 s) whose boxes overlap at IoU >= `DUP_IOU` = 0.3 at the earlier of the two frames. 0.3 is a generous screening level (one person under two ids gives IoU far above 0.5, two fighters in a clinch overlap partly); it comes from the review brief and was not tuned on the Olympic numbers. Reported, never subtracted | two ids on one person (4 px apart, IoU 0.87): 2 bursts, 1 pair; two separate people bursting together: 0; 20 px shift (IoU 0.47) counts, 35 px (0.23) does not; a twin 5 frames later counts, 6 does not; an explained twin does not count |
| N2 tests | Nothing protected pooling or the summed exposure of overlapping fragments (a mean of per-video rates, or a union exposure, passed the one-fragment fixture) | tests only | videos of 2 unexplained / 1.0 min (2.0 per min) and 18 / 3.0 min (6.0 per min) pool to 5.0, not the mean 4.0; own labels 3 / 1.0 min and 3 / 3.0 min pool to 1.5, not 2.0; two fragments on screen for the same 400 frames keep 800 exposure frames, not the union's 400, and the rates divide by 800 |

The CSV was NOT changed (no `possible_duplicate` column); a sampled row's twin can be found by its frame and box.

**Existing assertions replaced (nothing weakened):** the two checks that asserted the words "UNDERCOUNT" / "decisive" in the json bias and
the CLI print now assert the corrected statement ("coverage can only lose", "DUPLICATE", "neither a bout-level upper nor a lower bound",
"weak evidence", no "UNDERCOUNT", "possible twin"), which is a stricter set of facts about the same text.

**Mutation check (change applied in a scratch copy; the named suite must fail)**
| # | Mutation | Suite | Result | First failing checks |
|---|---|---|---|---|
| 15 | B1: `find_bursts_credited` does not verify the attributions | credited | caught, 3 fail | swapped ownership refused; credit removed; ambiguity flipped |
| 16 | B1: `find_bursts` does not verify them | credited / linking | caught, 2 / 0 | `find_bursts` refuses swapped ownership; wrist changed (the linking suite never swaps tables) |
| 17 | B1: verification compares only whether a track is credited, not which | credited | caught, 4 | swapped ownership refused by both functions |
| 18 | N1: duplicate counter without the IoU check | credited | caught, 2 | two separate people are not duplicates; the threshold bites on both sides |
| 19 | N1: duplicate counter ignores the time window | credited | caught, 1 | 5 frames counts, 6 does not |
| 20 | N1: duplicate counter counts explained twins | credited | caught, 1 | an explained twin does not count |
| 21 | N2: pooled rate is the mean of per-video rates | credited | caught, 3 | 5.0 not 4.0; 1.5 not 2.0; the no-credited-punch run |
| 22 | N2: a video's exposure is the longest fragment, not the sum | credited | caught, 2 | 800 frames, not 400; rates over 800 |

**Real data, `--credited` and `--credited --min-punches 2`, re-run:** every pre-existing number is unchanged (the same fragments, bursts,
unexplained, exposure, labels, rates, and the two samples are byte-identical); only the new keys appear.
| Quantity | Credited, min 1 | Credited, min 2 |
|---|---|---|
| unexplained bursts / `possible_duplicates` (share) | 7,318 / 177 (2.4%) | 4,342 / 66 (1.5%) |
| `possible_duplicate_pairs` | 89 | 33 |
| videos with no possible duplicate | 6 of 29 | 16 of 29 |
| most in one video | 22 (GH089681), 16 (GH109681, GH069681) | 10 (the same three videos) |

These are SCREENING counts, not a bound: 177 of 7,318 unexplained bursts (2.4%, 89 candidate pairs) at min 1 and 66 of 4,342
(1.5%, 33 pairs) at min 2 are flagged as possible duplicates. Duplication outside the screen is unmeasured: one physical punch can give
peaks more than 0.1 s apart on two ids, boxes overlapping at IoU below 0.3, or a twin that was matched to a label, and none of those is
flagged. (Corrected after Astra 6's re-review: an earlier wording called 89 / 7,318 an upper bound on the inflation and concluded that
duplication could not explain the rate; the counter cannot establish either.) Whether the unexplained bursts are punches is for the sample to say.
**Tests (all suites):** `test_completeness_credited.py` 95 (was 75), `test_linking.py` 266, `test_tracks.py` 79, `test_olympic.py` 33,
`test_hands.py` 38, `test_labels.py` 50, `test_ingest.py` 124, `test_env.py` 36, `test_pipeline.py` 260.
**Residual risks added:** (1) `possible_duplicates` is a screen: boxes that overlap at IoU >= 0.3 and bursts within 0.1 s, not a proof of
one person, and a twin that was matched to a label is not counted; (2) the verification recomputes attribution with default parameters,
so attributions made with other parameters are refused rather than trusted.


