"""Check an LLM narrator's reply against the report's facts, and write the narrated report only if it passes.

  python cli/report_punches.py ... --prompt-out          # writes <clip>.narration-prompt.json
  (paste its "prompt_text" into a model, or send "system" + "user"; save the reply as draft.json)
  python cli/narration_check.py --prompt results/reports/<clip>.narration-prompt.json --draft draft.json

Writes <clip>.narrated.txt beside the prompt when the draft keeps every rule (boxingvi.narrate.check): numbers
only from the facts each sentence cites, none spelled out, no absolutes, no failed fact in the headline, any
sentence relying on one hedged. Otherwise prints why and exits 1; the template summary stays the report.
Never sends anything anywhere. A prompt marked may_leave_machine: false holds participant-derived numbers: use a
local model only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boxingvi import boxers, narrate  # noqa: E402

MAX_PROMPT_BYTES = 1024 * 1024
SUFFIX = ".narration-prompt.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", type=Path, required=True, help=f"<clip>{SUFFIX} from report_punches.py --prompt-out")
    ap.add_argument("--draft", type=Path, required=True, help="the model's reply, saved as a file")
    a = ap.parse_args(argv)
    if not a.prompt.name.endswith(SUFFIX):
        raise ValueError(f"--prompt must be a <clip>{SUFFIX} file")
    raw = boxers._read_capped(a.prompt, MAX_PROMPT_BYTES, "narration prompt")
    prompt = boxers._json_object(raw, a.prompt.name)
    draft = boxers._read_capped(a.draft, narrate.MAX_DRAFT_BYTES, "draft")
    ok, problems, text = narrate.check(prompt, draft)
    if not ok:
        print("draft refused; the template summary stays the report:")
        for p in problems:
            print(f"  - {p}")
        return 1
    out = a.prompt.with_name(a.prompt.name[:-len(SUFFIX)] + ".narrated.txt")
    boxers.replace_bytes(out, (text + "\n").encode("utf-8"))
    print(text)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
