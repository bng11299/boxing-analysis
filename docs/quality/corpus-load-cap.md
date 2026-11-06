## `fix/corpus-load-cap` -- 2026-10-20

`data.py` read the BoxingVI corpus with `np.load` at four sites (`SKELETON_READERS['.npy']`,
`_load_presegmented`, `_load_v6`, `load_v6_track`). The corpus is third-party data and this
repo's project rules name the loader as the trust boundary. Follow-up from `fix/pose-load-cap`
(PR #24), on whose branch this is stacked, reusing its header-first approach.

Acceptance criteria: (1) a ~128-byte corpus `.npy` whose header claims 10^12 or 2M frames is
a ValueError naming the size check, with a traced peak under 5 MB, through every entry point;
(2) wrong dtype, wrong shape and object dtype are each a ValueError naming that check, raised
before any array is allocated, and an object array is never unpickled; (3) real-shaped arrays
(and all ten real files) load identical to `np.load`; (4) `pose_yolo` behaves as before
(`test_ingest` 159); (5) all suites green in a clean clone.

Threat sketch. Assets: the memory of the machine (training runs share it) and the integrity
of every clip the classifier and punch finder read. Entry points: `load_corpus`,
`load_v6_track` (`run_baseline`, `run_punch_finder`, `run_punch_extent`,
`eval_classify_v6`, `train_classifier`), `SKELETON_READERS`. Trust boundary: `data/`,
downloaded from a third party's Drive folder. Top threats: a small crafted file forcing a
huge allocation; a pickled array executing code (line 157 relied on numpy's
`allow_pickle=False` default rather than saying so).

Real-world facts checked before building: the bomb reproduced on the old loaders. A 128-byte
`V6.npy` claiming 2M frames allocated 816 MB, then raised an ordinary-looking ValueError
("cannot reshape array of size 0"); 10^12 frames was a MemoryError. New beyond PR #24: on a
plain file, a 112-byte `.npy` whose format-2.0 header claims a 4 GB header length made
`np.load` allocate 4.3 GB (Python's buffered `read(n)` reserves n bytes first), and 300
nested parentheses escape `np.load` as `tokenize.TokenError`. All ten real files share two
layouts, read from their headers: nine `(N, 25, 17, 2)` and V6 `(46497, 1, 17, 3)`, all
`<f8`, format 1.0. So the strict rules refuse nothing real.

### Iteration 1
- **Build:** `load_corpus_npy(path, layouts)` reads the header through the byte cap, then
  checks the dtype (plain float of 8 bytes or less; object, structured, subarray, integer,
  bool, complex and text refused), integer dims matching a named layout (`clips`
  `(N, 25, 17, 2 or 3)` or `v6` `(F, 1, 17, 3)`), and that the promised bytes fit in the file
  after the header (`fstat` on the open handle, exact Python-int product). Only then is one
  buffer of that size filled. The four call sites use it; their own post-load shape checks
  are now redundant and removed. The capped header reader and the chunked data reader move
  from `pose_yolo` to `npyread.py`, shared by both (`data` cannot import `pose_yolo`:
  `pose_yolo` imports `data`). `pose_yolo`'s messages are byte-identical.
- **Test:** `test_pipeline` 260 -> 311 (51 new): bombs through all four entry points, a
  one-byte-short file, the 4 GB header-length claim, an unparseable header, bad magic; five
  wrong dtypes plus structured; nine wrong shapes including V6's layout given to the clip
  loader and vice versa, and a `(True, ...)` dim; object arrays through all four entry points
  with a pickle probe that records if it is ever unpickled (the fixture is checked to fire
  under `np.load(allow_pickle=True)`); six real-shaped layouts (release `<f8`, float32
  3-channel, Fortran order, big-endian, zero clips, V6) identical to `np.load`; a peak no
  more than one array's size; trailing bytes ignored. Every bomb and order check asserts
  both the message and the traced peak, because the old loader also raised ValueError, just
  after allocating. Can the tests fail? Against the base commit's `data.py`, 37 of the 51
  fail (peaks of 680 MB, 816 MB and 4.3 GB). Six guards reverted one at a time (dtype,
  layout, int dims, size, header cap, TokenError conversion), and all six were caught.
  Other suites unchanged: ingest 159, classify 202, report 130, labels 50, env 36, hands 38,
  machine scripts 38; `run_baseline.py --synthetic` exits 0.
- **Real data:** all ten files load identical to `np.load` (values and dtype).
  `load_corpus(include_v6=True)` gives 5,442 clips with the same SHA-256 over every clip's
  source, label and keypoints as the base branch, and `load_v6_track` has the same digest.
- **Security (own pass, changed files):** no `allow_pickle` path exists; an allocation is
  bounded by the file's own size, taken by `fstat` on the handle that is read, so growth
  after the check is irrelevant and shrinkage is caught by the truncation guard. Errors name
  the basename, integer dims and the dtype string only.
- **Review (code-reviewer): FIX FIRST.** B1: numpy's header parser lets more than
  `TokenError` out of a malformed header, so the "anything inside it, ValueError" contract
  still broke: `'descr': ()` raised IndexError, a 4,000-term `1+1+...+1` shape raised
  RecursionError, and 9,000 minus signs raised MemoryError (a parser-stack overflow on a
  9 KB header, with nothing large allocated). `pose_yolo` shared the gap. B2: "allocates the
  array once" could not fail: on a 27 KB array its 0.5 MB slack passed `np.load(...)` copied
  twice. Non-blocking: dtype-check duplication between the two loaders; float16 accepted
  without saying so; big-endian returned as the file has it. The reviewer also fuzzed 20,000
  header-heavy byte mutations of three valid files: 19,561 ValueError, 438 loaded, 1
  UnicodeDecodeError (a ValueError), nothing else escaped, and no peak tied to a claimed size.

### Iteration 2
- **Fix:** `npyread.read_header` turns every non-ValueError from the capped header parse into
  ValueError("unparseable .npy header"). That is safe because the header is at most 10,012
  bytes, so these are malformed files and not the machine failing. B2, done properly, then
  failed on real behaviour: the reader peaked at 2x the array, because each chunk came back
  as a `bytes` object (up to 16 MB) and was then copied in. `read_data` now uses `readinto`
  straight into the array's buffer, so a plain file is read with no second copy. The
  docstring states that float16 is accepted and that byte order is left to the consumers,
  which all convert. Not taken: a shared dtype helper and moving the size bound into
  `read_data`. The two loaders bound sizes differently (zip ratio against file size) and
  check dtypes differently (per-key kinds, text length), so one helper would hide that.
- **Test:** `test_pipeline` 314 (54 new): the four malformed-header cases; the
  allocate-once check on a 6.8 MB array at 1.25x, with no slack (measured 6.8 MB now, 13.6 MB
  with the old reader). `test_ingest` 162 (3 new): the same three non-TokenError headers
  through `PoseSequence.load`. Can the tests fail? Against the base commit's `data.py`, 40 of
  the 54 fail. Eight guards were reverted one at a time (dtype, layout, int dims, size,
  header cap, error conversion narrowed back to TokenError, error conversion off, the old
  copying reader), and all eight were caught. Other suites unchanged: classify 202,
  report 130, labels 50, env 36, hands 38, machine scripts 38; `run_baseline.py --synthetic`
  exits 0.
- **Real data, again:** the ten corpus files are identical to `np.load`, and the 5,442
  clips and the V6 track give the same digests as the base branch. Because `readinto` also
  serves the pose path, the seven real pose files (`results/ingest/`) were rechecked: they
  are byte-identical to `np.load`. My first comparison said "DIFFERENT" for six of them,
  but that was NaN != NaN in undetected joints, and the base branch gives the same result.
- **Security (own re-pass):** the broadened `except` sits inside the header cap only;
  `read_data`'s truncation guard is unchanged in substance (`readinto` returning 0 or None).
- **Residual, accepted:** a genuine N-byte corpus file costs N bytes to load, as with
  `np.load`; there is no absolute cap because the release's largest file is 19 MB and a
  plain `.npy` cannot amplify. A FIFO planted in `Skeleton_data` would block a read on a
  POSIX machine, but planting one needs local write access to `data/`.
- **Reproduce:** a clean clone at 1ef2b9b (no `data/`, no `results/`, none tracked) gives the
  same numbers in every suite (pipeline 314, ingest 162, classify 202, report 130, labels 50,
  env 36, hands 38, machine scripts 38; `run_baseline.py --synthetic` exits 0). The pipeline
  and ingest check lists match the working tree's line for line.

