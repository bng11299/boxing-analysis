## `fix/pose-load-cap` -- 2026-10-20

`PoseSequence.load`, which every tool that reads a `.pose.npz` shares, called `np.load`, and
`np.load` allocates whatever shape a `.npy` header claims. Found in the 2026-10-19 review of
`render_punches.py` (PR #18), which converts the MemoryError at its own load site only; the
cap belongs in the loader. Stacked on `feat/output-report`.

Acceptance criteria: (1) a ~2 KB file whose headers claim 10^12 frames is a ValueError
naming the cap, with a traced peak under 5 MB; (2) a missing key, mismatched shapes and an
object dtype are each a ValueError naming that check, raised before any array is allocated;
(3) more than `MAX_POSE_FRAMES` (2,000,000, ~9 h at 60 fps, `video.MAX_TIMING_PACKETS`'s
bound) is refused and exactly the cap is not; (4) the seven real pose files load identical
to `np.load`, and save/load round-trips exactly; (5) all suites green in a clean clone.

Threat sketch. Assets: the memory of the machine running the tools (a training run may share
it), and the integrity of what every downstream tool reads. Entry points: `--pose` of
`find_punches`, `classify_punches`, `report_punches` (and `render_punches` on PR #18).
Trust boundary: files on disk between tools, which move between machines
(`docs/MACHINE-SYNC.md`). Top threats: a small crafted file forcing a huge allocation; a
pickled member executing code (already off, kept off by construction).

Real-world fact checked before building: the bomb reproduced on the old loader. A 1.8 KB
file claiming 10^12 frames attempted a 124 TiB allocation (MemoryError); one claiming 3M
frames allocated 408 MB, then failed on missing data. All seven real files
(`results/ingest/`) share one layout: xy/conf `<f4`, detected `|b1`, fps `<f8`, width/height
`<i8`, short `<U` text; the longest is V4 at 35,363 frames. So strict dtype rules refuse
nothing real.

### Iteration 1
- **Build:** the loader opens the file once and reads it as a zip. Every member's header is
  read with `read_magic` and `read_array_header_1_0`/`_2_0` and checked before anything is
  allocated: all eight keys present, dtype kind (xy/conf float, detected bool, fps numeric,
  width/height integer, source/model unicode up to 4,096 characters), shapes agreeing with
  `detected`'s (an int frame count, not `(True,)`), and the frame cap. Only then is data
  read, in 16 MB chunks, into one buffer of exactly the validated size, from the same open
  file. `allow_pickle` is never involved: object dtypes are refused from the header.
- **Test:** 147 checks in `test_ingest.py` (38 new). 9 guards reverted one at a time, and all
  9 were caught. The truncation guard was caught as a hang: without it the read loop never
  ends on short data. My own first test had a wrong needle: the old pickled-xy fixture does
  hold `xy`, so the loader rightly names the other seven keys.
- **Security (own pass, changed files):** two more ways past the header check, both measured.
  (a) zipfile decompresses bz2 and lzma members with no output limit, so reading the 8-byte
  magic of a 2.1 KB bz2 member expanded 584 MB. (b) Format 2.0 gives a header a 4-byte length,
  and numpy reads the whole claimed length before checking its own 10,000-byte limit. A
  263 KB deflated member reached 584 MB this way, about 1000:1. `np.load` had both. Fixed:
  only stored and deflate are accepted (what `np.savez` and `np.savez_compressed` write), and
  the header parser reads through a 10,012-byte cap. All three vectors now peak at 0.1 MB or
  less. Error messages name the file's basename only.
- **Review (code-reviewer): FIX FIRST.** B1: a header that `literal_eval` rejects makes numpy
  re-tokenize it, and `tokenize.TokenError` (from an unterminated header, or 300 nested
  parentheses) escaped the ValueError contract. Non-blocking: a 1.3 KB file whose headers
  claim exactly the cap at float64 passes the cap and commits 544 MB before finding the data
  missing; one `refuses` needle was loose; `NotImplementedError` is already a
  `RuntimeError`. The reviewer also fuzzed 40,000 random byte mutations of a stored and a
  deflated real save, and nothing escaped other than ValueError/OSError.

### Iteration 2
- **Fix:** header parse errors from tokenize become ValueError. Before `np.empty`, the bytes
  a header promises are compared with what its zip entry could hold: the stored size, or
  1032x the compressed size for deflate (deflate's ceiling), never more than the declared
  uncompressed size, and the compressed size never taken as more than the file's real size.
  The 544 MB case is now refused at 0.1 MB. The needle is tightened; the docstring says a
  file that cannot be opened is OSError and anything inside it ValueError.
- **Test:** 159 checks (50 new). The two new guards were reverted and both caught; 11 of 11
  overall. The reviewer's 33 probe files all either load (the legitimate ones) or are
  refused as ValueError. Re-fuzzed with 2x20,000 mutations on the final source: 0 escapes.
  Seven real files identical to `np.load` (values and dtypes). Other suites unchanged:
  pipeline 260, classify 202, report 130, labels 50, env 36, hands 38, machine scripts 38.
- **Review (own re-read of the final diff):** no new findings.
- **Reproduce:** a clean clone at d435520 (no `results/`, no `data/`) gives the same numbers
  in every suite, and the ingest check list matches the working tree line for line.
- **Residual, accepted:** a legitimate file at the cap still costs up to ~410 MB (float32)
  or ~820 MB (float64) to load; that is the cap's stated price. A zero-frame pose loads,
  as before. When this and PR #18 both land, `render_punches`' own MemoryError conversion
  becomes redundant but harmless.

