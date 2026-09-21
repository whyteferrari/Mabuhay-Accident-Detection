"""
A Python front end for vehicle_accident_detection.py.

Gradio, so the whole interface is Python - no HTML or JavaScript - but it
opens in a browser like a normal local site.

The app never ends a session on its own. Detecting an accident doesn't stop
anything; neither does the detector finishing, failing, or crashing. Each
of those just updates the page, and the upload box is immediately ready for
the next video. The only thing that shuts the app down is you stopping it
in the terminal.

    pip install -r requirements.txt
    python app.py           # opens http://127.0.0.1:7860

The detector needs six small edits first - see README.md.
"""

import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DETECTOR_NAME = "vehicle_accident_detection.py"

# app.py lives inside vehicle_detection_only, right next to the detector.
DETECTOR_SCRIPT = Path(os.environ.get("DETECTOR_SCRIPT", BASE_DIR / DETECTOR_NAME)).resolve()
RUNS_DIR = Path(os.environ.get("RUNS_DIR", BASE_DIR / "runs")).resolve()
RUNS_DIR.mkdir(parents=True, exist_ok=True)

POLL_SECONDS = 0.4
MAX_LOG_LINES = 400

# --- Lines worth pulling out of the detector's stdout -------------------
RE_PROGRESS = re.compile(r"^\[progress\]\s+(\d+)\s*/\s*(\d+)")
RE_ALERT = re.compile(r"^\[ALERT\]\s+(.*)$")
RE_DB_QUEUED = re.compile(r"^\[db\]\s+queued accident (\S+) for review \(track ([^,]+), severity=(\w+)\)")
RE_SNAPSHOT = re.compile(r"trigger snapshot saved:\s+(\S+)")
RE_RESCUE = re.compile(r"^\s*\[rescue\]\s+(.*)$")
RE_SEVERITY = re.compile(r"(CRITICAL|SEVERE|MODERATE|MILD)\s+ACCIDENT")
RE_TIME = re.compile(r"t=\s*([\d.]+)s")
RE_TRACKS = re.compile(r"\btracks?\s+([\d\s&]+)", re.I)

SEVERITY_MARK = {"critical": "Critical", "severe": "Severe", "moderate": "Moderate", "mild": "Mild"}


def track_key(text):
    """The track ids a line is about, order-independent.

    Every line the detector prints about one crash names the same ids -
    "Track 8", "Tracks 8 & 2", "track 8" - which is what lets the initial
    `[db] queued` line and the repeated `[ALERT]` lines collapse into a
    single row instead of five."""
    m = RE_TRACKS.search(text)
    if not m:
        return ()
    ids = re.findall(r"\d+", m.group(1))
    return tuple(sorted(ids, key=int))


def clock(seconds):
    if seconds is None:
        return "-"
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


class Run:
    """One detector subprocess and everything parsed out of its output.

    The detector is run as a separate process rather than imported. Its
    tracking state lives in module-level dicts that are never reset -
    canonical_id, involved_tracks, last_seen_pos_by_canonical - so a second
    video handled inside this same interpreter would inherit track ids and
    "already crashed" locks from the first. A fresh process per video gets a
    clean slate, and just as importantly, a crash in ultralytics or OpenCV
    takes down that process while this app carries on.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.id = None
        self.dir = None
        self.video_name = ""
        self.status = "idle"        # idle | running | done | failed | stopped
        self.frame = 0
        self.total = 0
        self.video_time = None
        self.log = []
        self.events = []            # dicts, newest last
        self.error = None
        self.proc = None
        self.started_at = None
        self.orphan_snapshots = []

    # -- lifecycle -----------------------------------------------------
    @property
    def is_running(self):
        return self.status == "running"

    def start(self, video_path):
        if not DETECTOR_SCRIPT.exists():
            raise FileNotFoundError(
                f"No detector at {DETECTOR_SCRIPT}. Make sure app.py is inside "
                f"vehicle_detection_only, next to {DETECTOR_NAME}, or set "
                f"DETECTOR_SCRIPT to its full path."
            )

        run_id = datetime.now().strftime("%H%M%S-") + uuid.uuid4().hex[:6]
        run_dir = RUNS_DIR / run_id
        (run_dir / "snapshots").mkdir(parents=True, exist_ok=True)

        src = Path(video_path)
        local = run_dir / ("input" + (src.suffix or ".mp4"))
        shutil.copy2(src, local)

        env = os.environ.copy()
        env.update({
            "VIDEO_PATH": str(local),
            "HEADLESS": "1",
            "OUTPUT_VIDEO_PATH": str(run_dir / "annotated.mp4"),
            "PREVIEW_PATH": str(run_dir / "preview.jpg"),
            "DEBUG_SNAPSHOT_DIR": str(run_dir / "snapshots"),
            "PYTHONUNBUFFERED": "1",
        })

        proc = subprocess.Popen(
            [sys.executable, "-u", str(DETECTOR_SCRIPT)],
            cwd=str(DETECTOR_SCRIPT.parent),   # so best_vehicle.pt and .env resolve as before
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with self.lock:
            self.reset()
            self.id = run_id
            self.dir = run_dir
            self.video_name = src.name
            self.status = "running"
            self.started_at = time.time()
            self.proc = proc

        threading.Thread(target=self._consume, args=(proc, run_dir), daemon=True).start()
        return run_id

    def stop(self):
        with self.lock:
            proc, running = self.proc, self.is_running
            if running:
                self.status = "stopped"
        if running and proc is not None:
            proc.terminate()

    # -- output parsing ------------------------------------------------
    def _consume(self, proc, run_dir):
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                with self.lock:
                    self._handle(line, run_dir)
        finally:
            proc.wait()
            with self.lock:
                self.proc = None
                if self.status == "stopped":
                    pass                      # the user asked for this
                elif proc.returncode == 0:
                    self.status = "done"
                    if self.total:
                        self.frame = self.total
                else:
                    self.status = "failed"
                    tail = [l for l in self.log[-30:] if l.strip()]
                    self.error = tail[-1] if tail else f"Detector exited with code {proc.returncode}."

    def _handle(self, line, run_dir):
        m = RE_PROGRESS.match(line)
        if m:
            self.frame, self.total = int(m.group(1)), int(m.group(2))
            return                            # progress lines are noise in the log pane

        self.log.append(line)
        if len(self.log) > MAX_LOG_LINES * 3:
            del self.log[:MAX_LOG_LINES]

        m = RE_TIME.search(line)
        if m:
            self.video_time = float(m.group(1))

        m = RE_ALERT.match(line)
        if m:
            self._add_event("Accident", m.group(1))
            return

        m = RE_DB_QUEUED.match(line)
        if m:
            self._add_event("Accident", f"Track {m.group(2)} logged as accident {m.group(1)}",
                            severity=m.group(3).lower())
            return

        m = RE_RESCUE.match(line)
        if m:
            self._add_event("Rescue pass", m.group(1))
            return

        m = RE_SNAPSHOT.search(line)
        if m:
            self._attach_snapshot(Path(m.group(1)).name)

    def _add_event(self, kind, text, severity=None):
        # ALERT lines carry no video time of their own, so fall back to the
        # most recent time seen on any line - the debug stream keeps that
        # within a frame or two of the alert.
        m = RE_TIME.search(text)
        at = float(m.group(1)) if m else self.video_time

        if severity is None:
            m = RE_SEVERITY.search(text)
            if m:
                severity = m.group(1).lower()

        ids = track_key(text)
        key = (kind,) + (ids or (text[:60],))

        for ev in self.events:
            if ev["key"] != key:
                continue
            ev["count"] += 1
            ev["text"] = text
            if severity:
                ev["severity"] = severity
            if at is not None:
                ev["at"] = ev["at"] if ev["at"] is not None else at
            return

        self.events.append({
            "key": key, "kind": kind, "text": text, "severity": severity,
            "at": at, "ids": ids, "count": 1, "snapshot": None,
        })
        self._claim_orphans()

    def _attach_snapshot(self, name):
        """Hang a debug snapshot on the event it belongs to. The detector
        names them for the tracks involved - track8_car_t12.30s.jpg,
        overlap_8_2_t12.90s.jpg - so the ids in the filename find the row.
        A snapshot can be written a frame before its alert prints, so
        anything unmatched is parked and picked up later.

        The filename also carries the trigger time to the hundredth of a
        second, which beats the fallback an ALERT line gets (the last time
        seen anywhere in the output), so it overrides it."""
        head, _, tail = name.partition("_t")
        ids = tuple(sorted(re.findall(r"\d+", head), key=int))
        m = re.match(r"([\d.]+)s", tail)
        at = float(m.group(1)) if m else None

        for ev in reversed(self.events):
            if ev["snapshot"] is None and ev["ids"] == ids:
                ev["snapshot"] = name
                if at is not None:
                    ev["at"] = at
                return
        self.orphan_snapshots.append((ids, name, at))

    def _claim_orphans(self):
        for entry in list(self.orphan_snapshots):
            ids, name, at = entry
            for ev in self.events:
                if ev["snapshot"] is None and ev["ids"] == ids:
                    ev["snapshot"] = name
                    if at is not None:
                        ev["at"] = at
                    self.orphan_snapshots.remove(entry)
                    break

    # -- what the page shows -------------------------------------------
    def snapshot(self):
        with self.lock:
            rows = []
            gallery = []
            for ev in self.events:
                rows.append([
                    clock(ev["at"]),
                    SEVERITY_MARK.get(ev["severity"], ev["kind"]),
                    " & ".join(ev["ids"]) or "-",
                    ev["count"],
                    ev["text"],
                ])
                if ev["snapshot"]:
                    path = self.dir / "snapshots" / ev["snapshot"]
                    if path.exists():
                        caption = f"{clock(ev['at'])} · {SEVERITY_MARK.get(ev['severity'], ev['kind'])}"
                        gallery.append((str(path), caption))
            return {
                "status": self.status,
                "video_name": self.video_name,
                "frame": self.frame,
                "total": self.total,
                "error": self.error,
                "rows": rows,
                "gallery": gallery,
                "log": "\n".join(self.log[-MAX_LOG_LINES:]),
                "preview": self.dir / "preview.jpg" if self.dir else None,
                "video": self.dir / "annotated.mp4" if self.dir else None,
                "elapsed": (time.time() - self.started_at) if self.started_at else 0.0,
            }


run = Run()


def read_preview(path):
    """Load the detector's current frame. It writes to a temp file and
    renames, so a half-written JPEG shouldn't be visible - but a torn read
    is still possible on some filesystems, and a dropped preview frame
    isn't worth interrupting a run over."""
    if not path or not Path(path).exists():
        return None
    try:
        with Image.open(path) as im:
            return im.convert("RGB").copy()
    except Exception:
        return None


def status_line(s):
    parts = []
    if s["video_name"]:
        parts.append(s["video_name"])
    if s["total"]:
        pct = 100 * s["frame"] / s["total"]
        parts.append(f"frame {s['frame']:,} of {s['total']:,} ({pct:.0f}%)")
    elif s["frame"]:
        parts.append(f"frame {s['frame']:,}")
    if s["elapsed"]:
        parts.append(f"{s['elapsed']:.0f}s elapsed")

    word = {
        "idle": "Waiting for a video",
        "running": "Processing",
        "done": "Finished",
        "failed": "Stopped on an error",
        "stopped": "Stopped by you",
    }[s["status"]]
    return f"### {word}\n" + ("  \n".join(parts) if parts else "Upload a video to begin.")


def view(s):
    """Everything the page needs, in output order."""
    n = len(s["rows"])
    found = (
        f"Nothing flagged yet." if not n and s["status"] == "running"
        else "Nothing was flagged in this video." if not n and s["status"] in ("done", "stopped")
        else "" if not n
        else f"{n} flagged {'event' if n == 1 else 'events'}."
    )
    video = str(s["video"]) if s["video"] and Path(s["video"]).exists() else None
    return (
        status_line(s),
        s["rows"],
        read_preview(s["preview"]),
        s["gallery"],
        found + (f"\n\n**{s['error']}**" if s["error"] else ""),
        s["log"],
        gr.update(value=video, visible=video is not None),
    )


def start(video_path, progress=gr.Progress(track_tqdm=False)):
    """Launch a run, then stream the page until the detector exits.

    Detecting an accident is not an exit condition - nothing here breaks out
    of the loop on an event. The loop ends only when the subprocess ends, and
    even then the app stays up and ready for another video.
    """
    if not video_path:
        s = run.snapshot()
        yield view({**s, "error": "Choose a video first."})
        return

    try:
        run.start(video_path)
    except Exception as e:
        s = run.snapshot()
        yield view({**s, "status": "failed", "error": str(e)})
        return

    while True:
        s = run.snapshot()
        yield view(s)
        if s["status"] != "running":
            break
        time.sleep(POLL_SECONDS)

    yield view(run.snapshot())


def stop():
    run.stop()
    return gr.update()


with gr.Blocks(title="Accident detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Accident detector\n"
        "Upload traffic footage and watch the detector work through it. "
        "Flagged collisions appear below as they're found; the page stays open "
        "afterwards so you can run another video."
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Traffic footage", sources=["upload"])
            with gr.Row():
                run_btn = gr.Button("Run the detector", variant="primary")
                stop_btn = gr.Button("Stop this run")
            status_md = gr.Markdown("### Waiting for a video\nUpload a video to begin.")
            annotated_out = gr.Video(label="Annotated video", visible=False, interactive=False)

        with gr.Column(scale=1):
            preview_img = gr.Image(label="Current frame", height=340, interactive=False)

    found_md = gr.Markdown("")
    events_df = gr.Dataframe(
        headers=["Time", "Severity", "Tracks", "Seen", "What the detector reported"],
        datatype=["str", "str", "str", "number", "str"],
        col_count=(5, "fixed"),
        wrap=True,
        interactive=False,
        label="Flagged events",
    )
    gallery = gr.Gallery(label="Trigger frames", columns=4, height=260, object_fit="cover")

    with gr.Accordion("Detector output", open=False):
        log_box = gr.Textbox(lines=18, max_lines=18, show_label=False, interactive=False)

    outputs = [status_md, events_df, preview_img, gallery, found_md, log_box, annotated_out]
    run_btn.click(start, inputs=[video_in], outputs=outputs)
    stop_btn.click(stop, inputs=None, outputs=None)


if __name__ == "__main__":
    print(f"Detector: {DETECTOR_SCRIPT}" + ("" if DETECTOR_SCRIPT.exists() else "  (NOT FOUND)"))
    print(f"Runs:     {RUNS_DIR}")
    demo.queue().launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        inbrowser=True,
        show_api=False,
    )