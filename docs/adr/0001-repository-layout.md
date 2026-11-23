# ADR-0001: Runnable scripts move from the repo root to `cli/`

- Status: accepted
- Date: 2026-11-22
- Branch: `chore/repo-layout`
- Scope: this repo's layout only. The project's research decisions stay in the vault's ADR series
  (`03-Projects/PRJ-001-comps-topic-selection/`), so this numbering is separate.

## Context

By November 2026 the repo root held 36 runnable Python scripts next to the README, licence and
requirements files. The importable library was already in `src/boxingvi/`, and tests, docs, env tools and
derived results each had their own folder. The root was the only place without order: a reader who
opens the repo (the poster's QR code points at it) sees a long list of scripts before the README.

The scripts are not independent. They import each other by module name (`analyse_clip` imports
`find_punches`, `classify_punches`, `run_ingest` ...; every `olympic_*` script imports `olympic_link`), and
the tests import them the same way. Six scripts derive the repo root, and so `results/`, from
`__file__`.

Guidance read (2026-11-22):

- PyPA, [src layout vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/):
  importable code goes under `src/`; the src layout stops accidental use of the in-development copy.
  This repo already follows it.
- Scientific Python, [Simple packaging](https://learn.scientific-python.org/development/guides/packaging-simple/):
  `src/<package>/`, `tests/`, `docs/`, a `pyproject.toml`, and command-line tools as `[project.scripts]`
  entry points.
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/): data, models and
  reports in their own folders; `README`, `LICENSE` and `requirements.txt` at the root; code in one package.
- GitHub, [About CITATION files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files):
  a `CITATION.cff` at the root gives a "Cite this repository" link.

## Options

1. **Leave the root as it is.** No cost and no benefit.
2. **Move the scripts into one flat folder, `cli/`.** The cross-imports keep working, because Python puts
   a script's own folder on `sys.path`. Each move is a `git mv`; the edits are mechanical.
3. **Group the scripts by area** (`cli/olympic/`, `cli/eval/` ...). The groups import across each other
   (`olympic_ring` imports `find_punches`), so this needs package imports or `sys.path` edits per group.
4. **Move the logic into `src/boxingvi/` and expose thin `[project.scripts]` entry points** from a
   `pyproject.toml`. This is the end state that the Scientific Python guide describes. It is a refactor of
   the scripts' 9,800 lines, not a move, and it changes the install procedure on two machines.

## Decision

Option 2. Also add `CITATION.cff` (it cites BoxingVI, which is part of the terms of use) and a
"Repository layout" section in the README.

Not moved, on purpose:

| Path | Why it stays |
|---|---|
| `results/`, `weights/`, `data/` | Gitignored. `docs/RESULTS.md` and the reproduction commands name them; renaming them to Cookiecutter's `reports/` / `models/` would break reproduction and buys nothing. |
| `requirements*.txt` | The root is where Cookiecutter and pip users look; `scripts/setup-env.ps1` and the tests read them there. |
| `scripts/` | Environment and machine tooling (PowerShell + `check_env.py`), a different kind of thing from the analysis scripts. |
| `config/`, `labelstudio/`, `.streamlit/` | Small. Streamlit finds `.streamlit/` from the working folder. A live branch adds a file to `labelstudio/`. |
| `docs/quality/*` | Records of past quality-loop runs. They name scripts as they were at that commit, so they stay as written. Use the table below to map them. |

## Path changes

Every row is `<name>.py` → `cli/<name>.py`, by `git mv`:

`analyse_clip`, `blur_faces`, `boxer_punches`, `classify_punches`, `demo_app`, `eval_classify_v6`,
`eval_fast_combinations`, `eval_pipeline_v5v9`, `eval_pose_switch`, `find_punches`, `glove_identity`,
`import_labels`, `label_moments`, `learning_curve`, `linker_join_check`, `olympic_attribute`, `olympic_audit`,
`olympic_boxers`, `olympic_colour`, `olympic_completeness`, `olympic_framechange`, `olympic_headgear`,
`olympic_link`, `olympic_ring`, `pose_watcher`, `render_punches`, `report_boxers`, `report_punches`,
`run_baseline`, `run_ingest`, `run_punch_extent`, `run_punch_finder`, `score_combinations`, `train_classifier`,
`tune_fast_combinations`, `wait_for_pose`.

That is all 36 tracked `.py` files that were at the root. `git show --stat` of the move commit is the
authoritative list.

Commands change from `python run_baseline.py ...` to `python cli/run_baseline.py ...`, still run from the
repo root. Outputs still land in `results/` at the repo root: each script's repo-root anchor gained one
`.parent`.

## Consequences

- **The results freeze is unaffected.** `results-freeze-2026-11-06` is a tag on an older commit; check it
  out and the old commands in its `docs/RESULTS.md` work as written. On this branch the same commands take
  the `cli/` prefix and write to the same `results/` paths.
- **GPU laptop:** the pose-watcher scheduled task stores the full path to `pose_watcher.py`. After pulling
  this change, run `.\scripts\install-pose-watcher.ps1` again, or the task starts a file that is gone.
- **Glove-identity gate (N2):** `glove_identity.py` hashes its own path and contents into the code digest,
  and checks `git diff <frozen commit> -- src/boxingvi glove_identity.py`. Its path is now
  `cli/glove_identity.py`. N2 was scored once and is closed; a re-score after this commit would refuse,
  which is the gate working as designed (any edit under `src/boxingvi/` does the same).
- **Open branches** that add or edit root scripts (`feat/scouting-summary`, `codex/sparring-labels`) need a
  rebase; Git's rename detection carries edits across, but new scripts must be put in `cli/` by hand.
- **Deferred:** a `pyproject.toml` with `[project.scripts]` (option 4). Do it when the logic moves into the
  library, not as part of a move.
