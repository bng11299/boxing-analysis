"""Local demo of analyse_clip.py: a thin Streamlit page over src/boxingvi/demo.py.

  python -m streamlit run demo_app.py -- --results-dir results/analyse [--uploads-dir results/demo_uploads]

Run it from the repo root (Streamlit reads .streamlit/config.toml from there). It binds to 127.0.0.1,
sends no telemetry and loads nothing from outside this laptop. Two tabs: a Gallery of finished clips
(instant, safe for a live demo) and Analyse a clip (upload, run analyse_clip.py, show the same result).

Every decision about what is trusted -- paths, names, argument lists -- is in demo.py and tested in
tests/test_demo.py; this file only draws widgets. Own footage only: uploads and their outputs live
under results/ (gitignored) and can be deleted from the page.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from boxingvi import demo  # noqa: E402

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
LOG_LINES_KEPT = 500
LOG_REFRESH_S = 0.4


def parse_cli() -> argparse.Namespace:
    """Flags after `--` on the streamlit command line. A mistyped flag is an error, not ignored."""
    ap = argparse.ArgumentParser(prog="streamlit run demo_app.py --", allow_abbrev=False)
    ap.add_argument("--results-dir", type=Path, default=demo.DEFAULT_RESULTS,
                    help="where finished clips are listed (default %(default)s)")
    ap.add_argument("--uploads-dir", type=Path, default=demo.DEFAULT_UPLOADS,
                    help="where uploads are stored; outputs go in <uploads-dir>/out (default %(default)s)")
    return ap.parse_args(sys.argv[1:])


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def show_result(clip: demo.Clip, key: str) -> None:
    """One finished clip: overlay video, the summary verbatim, the counts, the table, the downloads."""
    overlay = clip.path("overlay.mp4")
    st.subheader(clip.name)

    video_col, text_col = st.columns([2, 3], gap="large")
    with video_col:
        web = None
        try:
            with st.spinner("Preparing a web preview of the overlay (first view only)..."):
                web = demo.ensure_web_preview(overlay)
        except demo.DemoError as e:
            st.warning(f"{e}. The original overlay can still be downloaded below.")
        if web is not None:
            st.video(str(web))
        elif demo.find_ffmpeg() is None:
            st.warning("ffmpeg is not installed, so no web preview can be made. "
                       "The original overlay (large) can be downloaded instead.")
        if web is None:
            size_mb = overlay.stat().st_size / (1024 * 1024)
            st.download_button(f"Download the original overlay ({size_mb:.0f} MB)", data=overlay.open("rb"),
                               file_name=overlay.name, mime="video/mp4", key=f"{key}-overlay")
    with text_col:
        st.markdown("**Summary** (exactly as the tool wrote it, caveats included)")
        try:
            st.text(demo.read_summary(clip.path("summary.txt")))
        except (demo.DemoError, OSError) as e:
            st.error(f"Could not show the summary: {e}")

    try:
        punches = demo.summarise_punches(clip.path("classified.csv"))
    except (demo.DemoError, OSError) as e:
        st.error(f"Could not read the punches: {e}")
    else:
        left, right = st.columns([2, 3])
        with left:
            st.markdown(f"**Punches by type** ({punches.total} in all)")
            if punches.counts:
                chart = pd.DataFrame({"punch type": list(punches.counts), "punches": list(punches.counts.values())})
                st.bar_chart(chart, x="punch type", y="punches", sort="-punches", horizontal=True)
            else:
                st.info("No punches were found in this clip.")
        with right:
            st.markdown("**Each punch**")
            table = pd.DataFrame(punches.rows, columns=list(demo.TABLE_COLUMNS))
            for col in ("time_s", "confidence"):
                table[col] = pd.to_numeric(table[col], errors="coerce")
            st.dataframe(table, hide_index=True, height=360)

    d1, d2, _ = st.columns([1, 1, 3])
    with d1:
        st.download_button("Download report.xlsx", data=read_bytes(clip.path("report.xlsx")),
                           file_name=clip.path("report.xlsx").name, mime=XLSX_MIME, key=f"{key}-xlsx")
    with d2:
        st.download_button("Download classified.csv", data=read_bytes(clip.path("classified.csv")),
                           file_name=clip.path("classified.csv").name, mime="text/csv", key=f"{key}-csv")


def gallery_tab(results_dir: Path) -> None:
    try:
        clips = demo.find_clips(results_dir)
    except demo.DemoError as e:
        st.error(str(e))
        return
    st.caption(f"Finished clips in {results_dir}. A clip is listed only when all seven outputs are there.")
    if not clips:
        st.info("No finished clips here yet. Analyse one in the next tab, or point --results-dir at a folder that has them.")
        return
    names = [c.name for c in clips]
    chosen = st.selectbox("Clip", names, key="gallery-clip")
    show_result(clips[names.index(chosen)], key=f"gallery-{chosen}")


def run_upload(uploaded, stance: str, device: str, rounds: "str | None", uploads_dir: Path) -> None:
    """Store the upload, run analyse_clip.py on it, keep the log; the result is drawn by the caller."""
    st.session_state.pop("run_error", None)
    try:
        uploaded.seek(0)
        stored = demo.save_upload(uploads_dir, uploaded.name, uploaded)
        out_dir = demo.out_dir_for(uploads_dir)
        argv = demo.build_analyse_argv(stored, stance, device, rounds, out_dir, cuda_available=device != "cpu")
    except (demo.DemoError, OSError) as e:
        st.session_state["run_error"] = str(e)
        return
    st.session_state["upload_name"] = stored.stem
    lines: list[str] = []
    st.session_state["upload_log"] = lines
    with st.expander("Run log (analyse_clip.py)", expanded=True):
        box = st.empty()
        last = [0.0]

        def on_line(line: str) -> None:
            lines.append(line)
            del lines[:-LOG_LINES_KEPT]
            if time.monotonic() - last[0] > LOG_REFRESH_S:
                last[0] = time.monotonic()
                box.code("\n".join(lines), language=None)

        with st.spinner("Analysing: pose extraction on a CPU takes minutes per minute of video..."):
            rc = demo.run_analyse(argv, on_line)
        box.code("\n".join(lines), language=None)
    if rc != 0:
        st.session_state["run_error"] = (f"analyse_clip.py stopped (exit code {rc}). Nothing was published; "
                                         "the log above says which stage failed.")


def analyse_tab(uploads_dir: Path) -> None:
    out_dir = demo.out_dir_for(uploads_dir)
    st.warning("**This is slow on a CPU:** pose extraction takes minutes per minute of video, so a long clip "
               "is not a live-demo job. Use the Gallery for that. Stay on this tab while it runs.")
    st.caption("Own footage only. The clip is stored under the uploads folder with a generated name, "
               "never the name you gave it, and can be deleted below.")

    uploaded = st.file_uploader("Video (.mp4, .mov or .m4v)", type=[e.lstrip(".") for e in demo.ALLOWED_EXTENSIONS])
    c1, c2, c3 = st.columns(3)
    with c1:
        stance = st.selectbox("Boxer's stance (required: the tool never guesses)", list(demo.STANCES), index=None,
                              placeholder="Choose a stance")
    with c2:
        device = st.selectbox("Device", demo.available_devices())
    with c3:
        rounds_text = st.text_input("Rounds in seconds (optional)", placeholder="0-180,240-420")
    consent = st.checkbox("This is my own footage, or I have the filmed person's consent")

    rounds, rounds_error = None, None
    try:
        rounds = demo.validate_rounds(rounds_text)
    except demo.DemoError as e:
        rounds_error = str(e)
        st.error(rounds_error)

    ready = uploaded is not None and stance is not None and consent and rounds_error is None
    missing = [m for ok, m in ((uploaded is not None, "a video"), (stance is not None, "a stance"),
                               (consent, "the own-footage confirmation")) if not ok]
    if not ready and rounds_error is None and missing:
        st.caption("Still needed: " + ", ".join(missing) + ".")
    if st.button("Analyse this clip", type="primary", disabled=not ready):
        run_upload(uploaded, stance, device, rounds, uploads_dir)

    if err := st.session_state.get("run_error"):
        st.error(err)

    name = st.session_state.get("upload_name")
    if name:
        done = [c for c in demo.find_clips(out_dir) if c.name == name]
        if done:
            st.success("Done.")
            show_result(done[0], key=f"upload-{name}")
        elif not st.session_state.get("run_error") and st.session_state.get("upload_log"):
            with st.expander("Run log (analyse_clip.py)"):
                st.code("\n".join(st.session_state["upload_log"]), language=None)
        if st.button("Delete this upload and its outputs", key="delete-current"):
            remove_upload(uploads_dir, name)
            st.rerun()

    earlier = [p for p in demo.list_uploads(uploads_dir) if p.stem != name]
    if earlier:
        with st.expander(f"Earlier uploads on this laptop ({len(earlier)})"):
            for p in earlier:
                a, b = st.columns([4, 1])
                a.write(f"{p.stem}  ({p.stat().st_size / (1024 * 1024):.1f} MB)")
                if b.button("Delete", key=f"delete-{p.stem}"):
                    remove_upload(uploads_dir, p.stem)
                    st.rerun()


def remove_upload(uploads_dir: Path, name: str) -> None:
    try:
        removed = demo.delete_upload(uploads_dir, name)
    except (demo.DemoError, OSError) as e:
        st.session_state["run_error"] = f"Could not delete: {e}"
        return
    if st.session_state.get("upload_name") == name:
        for k in ("upload_name", "upload_log", "run_error"):
            st.session_state.pop(k, None)
    st.toast(f"Deleted {len(removed)} file(s) for {name}.")


def main() -> None:
    st.set_page_config(page_title="Boxing punch analysis", layout="wide")
    try:
        args = parse_cli()
        results_dir = demo.check_folder(args.results_dir)
        uploads_dir = demo.check_folder(args.uploads_dir)
    except SystemExit:
        st.error("Bad command line. Use: python -m streamlit run demo_app.py -- --results-dir <folder> "
                 "[--uploads-dir <folder>]")
        st.stop()
    except demo.DemoError as e:
        st.error(str(e))
        st.stop()

    st.title("Single-camera boxing analysis")
    st.caption("Punches found, named and counted from one phone video. Runs on this laptop only; "
               "nothing is uploaded anywhere.")
    gallery, analyse = st.tabs(["Gallery", "Analyse a clip"])
    with gallery:
        gallery_tab(results_dir)
    with analyse:
        analyse_tab(uploads_dir)


main()
