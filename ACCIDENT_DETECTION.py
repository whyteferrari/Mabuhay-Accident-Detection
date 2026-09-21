import cv2
import io
import math
import os
import threading
import numpy as np
from PIL import Image
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from ultralytics import YOLO
from supabase import create_client
from dotenv import load_dotenv

# Reads a ".env" file placed in the same folder as this script (see
# ".env.example" for the format) and loads its values into
# os.environ, so you don't have to `export` them by hand every time.
# If no .env file exists, this line just does nothing - normal
# exported environment variables still work as a fallback.
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

VEHICLE_MODEL_PATH = "best_vehicle.pt" # CHANGE PATH TO VEHICLE MODEL
ACCIDENT_MODEL_PATH = "best_accident_Aug28.pt" # CHANGE PATH TO ACCIDENT MODEL
# If either dedicated model above is missing, fall back to this single
# combined model instead of crashing on startup.
FALLBACK_MODEL_PATH = "best_accident_Aug28.pt"
VIDEO_PATH = "ScreenRecording_06-16-2026 00-59-54_1.mov" # CHANGE PATH TO VIDEO

CONFIDENCE = 0.001 #CONFIDENCE FOR VEHICLE DETECTION
ACCIDENT_CONFIDENCE = 0.01 #CONFIDENCE FOR ACCIDENT DETECTION.
IMAGE_SIZE = 640

# NMS IoU threshold for the vehicle detector. Ultralytics discards the
# lower-confidence box whenever two same-class detections overlap more
# than this. Left at the library default (0.7) that is a real problem
# during a collision: two vehicles' axis-aligned boxes can legitimately
# overlap 50-70%+ right as/after they hit each other - precisely the
# frame you need BOTH boxes for - and NMS will silently drop the
# weaker one, leaving one car with no box (and no track_id) at all.
# Raising this makes NMS only merge near-duplicate boxes, not two
# genuinely separate, heavily-overlapping vehicles.
NMS_IOU_THRESHOLD = 0.70

# --- Low-confidence "rescue" pass ---
# Some vehicles (dark colors, harsh shadow, odd angle after impact)
# simply score below CONFIDENCE on the main pass even though NMS never
# touched them. Rather than lowering CONFIDENCE globally - which would
# increase false-positive/flicker risk for the WHOLE video, not just
# crash moments - only relax the threshold in the narrow situation
# where it actually matters: right as/after some other nearby vehicle
# registers a SUDDEN STOP, re-run detection on just a local crop
# around that vehicle at a much lower confidence, to try to recover a
# second vehicle (e.g. the one that hit it) that the main pass missed
# entirely. This crop is drawn purely for the OVERLAP/PROXIMITY crash
# logic - it never creates its own independently-tracked ID, it only
# feeds compute_iou / compute_edge_distance against tracks that DID
# survive the main pass.
RESCUE_CONFIDENCE = 0.20
RESCUE_SEARCH_RADIUS_RATIO = 1.5  # how much wider/taller than the sudden-stop box to search, as a multiple of its own size

DEBUG_PRINT_STATUS_CHANGES = True  # prints every status change per track, with the accumulated stopped duration - turn off once things look right

# --- Supabase ---
# Use the SERVICE ROLE key here, never the anon key - this script is
# a trusted backend process, not a browser. Set these as environment
# variables rather than hardcoding them in the file:
#   export SUPABASE_URL="https://xxxx.supabase.co"
#   export SUPABASE_SERVICE_KEY="eyJ...service_role_key..."
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)

# Must match a row already in your `cameras` table (see schema.sql
# seed data). Update these to the real camera this video corresponds to.
CAMERA_ID = os.environ.get("CAMERA_ID")  # uuid string, or None
CAMERA_LOCATION = os.environ.get("CAMERA_LOCATION", "TEST CAMERA - TEST LOCATION aldryne test")

# Video files have no real-world start time by themselves - anchor
# frame_idx/fps (video-relative seconds) to an actual wall-clock
# moment so `accidents.timestamp` means something real. For a live
# RTSP feed instead of a recorded file, set this to
# datetime.now(timezone.utc) right before the capture loop starts.
VIDEO_START_WALLTIME = datetime.now(timezone.utc)

STORAGE_BUCKET = "accident-media"

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_ENABLED else None
if not SUPABASE_ENABLED:
    print("[warn] SUPABASE_URL / SUPABASE_SERVICE_KEY not set - accidents will be "
          "detected and shown locally but NOT written to the database.")

# --- Speed estimation calibration ---
# Approximate real-world width (meters) per class - used to derive a
# LOCAL pixels-per-meter scale from each vehicle's own box width every
# frame, instead of one fixed constant for the whole image.
VEHICLE_REAL_WIDTH_M = {
    "car": 1.8,
    "van": 2.0,
    "bus": 2.5,
    "jeepney": 2.0,
    "tricycle": 1.4,
    "motorcycle": 0.8,
    "truck": 2.3,
}
FALLBACK_PIXELS_PER_METER = 8.0

HISTORY_SECONDS = 1.0
MAX_HISTORY = 30
POSITION_SMOOTHING_FRAMES = 3

# --- "Stopped" detection ---
STOPPED_SPEED_KMH = 2.0
STOPPED_FRAMES_REQUIRED = 10
MOVING_FRAMES_TO_RESET = 10  # consecutive above-threshold frames needed before we give up on an accumulating stop - absorbs brief jitter/occlusion blips instead of resetting on one noisy frame

# --- "Sudden stop" detection ---
SUDDEN_STOP_DROP_KMH = 15.0
SUDDEN_STOP_WINDOW = 0.75

# --- Track trust ---
MIN_TRACK_AGE_FRAMES = 15

# --- ID-switch stitching ---
# The tracker (ByteTrack, or BoT-SORT without ReID) matches boxes
# frame-to-frame mostly by motion/IoU continuity. That's exactly what
# breaks at the moment of a collision: the box shape suddenly deforms,
# one vehicle may briefly occlude the other, and detection confidence
# can dip for a frame or two. When that happens the tracker doesn't
# "lose and resume" the same ID - it silently drops the old track and
# hands the vehicle a brand-new track_id a few frames later. Every
# per-track dict in this script (track_history, stopped_since,
# involved_tracks, db_accident_id, etc.) is keyed by track_id, so an
# ID switch mid-crash makes the vehicle look like two different
# vehicles: a "MOVING" one that vanishes, and a brand-new "SUDDEN
# STOP" one that appears out of nowhere right next to the vehicle it
# hit. That's what produced e.g. "8 & 2" instead of "8 & 1" - the
# vehicle that actually hit ID 8 WAS an earlier ID (say ID 1), but its
# track died on impact and got reborn under whatever the next free ID
# happened to be.
#
# This stitches that back together in Python, independent of whatever
# the underlying tracker does: when a brand-new raw track_id shows up,
# check whether some other track_id went missing very recently very
# close to where this new one first appears. If so, treat the new
# raw_id as a continuation of the old one (same canonical track_id),
# so all the history/state above stays attached to the vehicle, not
# to the accident-of-the-moment ID churn.
STITCH_MAX_FRAMES_GAP = 15     # how many frames a track can vanish for and still be re-matched (tune to ~0.5s at your fps)
STITCH_MAX_DIST_PX = 60        # how close the new track's first centroid must be to the old one's last known centroid
STITCH_DEBUG = True            # if True, logs WHY a brand-new raw id did/didn't get stitched, incl. near-miss
                                # candidates that failed distance/gap/class checks - turn on to diagnose a specific
                                # missed stitch, off once you've tuned the thresholds (it's noisy - logs on every
                                # new raw track, including totally unrelated background vehicles)

# --- "Sudden stop" detection (continued) ---
SUDDEN_STOP_DROP_KMH = SUDDEN_STOP_DROP_KMH  # (unchanged, kept here only as an anchor comment)

# --- Compound "likely accident" signal (motion only) ---
SUDDEN_STOP_TO_ACCIDENT_WINDOW = 5.0
STOPPED_DURATION_FOR_ACCIDENT = 2.0

# A vehicle that was moving fast right before it suddenly stopped is
# much more likely to have crashed than one that was already crawling.
# If the speed just before the sudden stop was at or above this, treat
# it as strong evidence on its own and require far less "stopped time"
# afterward before flagging LIKELY ACCIDENT - e.g. a drop from 40 km/h
# to a dead stop shouldn't need the same 2-second wait as a jeepney
# easing to a stop at a curb.
HIGH_SPEED_SUDDEN_STOP_KMH = 40.0
STOPPED_DURATION_FOR_ACCIDENT_HIGH_SPEED = 0.5

# --- Compound "likely accident" signal (box-overlap only) ---
# Two vehicle boxes overlapping heavily is itself suspicious - two
# vehicles occupying the same space almost never happens except at a
# collision (as opposed to just being near each other in adjacent
# lanes). This is intentionally independent of the speed/stop logic
# above: a T-bone or side-swipe can happen without either vehicle ever
# registering a clean "sudden stop."
OVERLAP_IOU_THRESHOLD = 0.05        # intersection-over-union between two boxes to count as "overlapping"
OVERLAP_FRAMES_REQUIRED = 5         # consecutive overlapping frames before it's trusted (debounces momentary bbox jitter as cars pass each other)
OVERLAP_DECAY_PER_MISSED_FRAME = 2  # how fast the counter drains on a frame where the pair no longer overlaps

# --- Compound "likely accident" signal (proximity + sudden-stop) ---
# A faster, more sensitive companion to the box-overlap signal above.
# Instead of requiring heavy box overlap (IoU), this only requires the
# box EDGES to be close together - touching or nearly touching, not
# necessarily overlapping - at the same moment at least one of the two
# vehicles registers a SUDDEN STOP. "Something braked hard right as it
# got close to another vehicle" is strong evidence on its own, so it
# only needs a couple of consecutive frames to trust, unlike IoU
# overlap which needs several seconds to rule out cars simply passing
# close together. This runs independently of, and in addition to, the
# IoU trigger - either one is enough to start an event.
PROXIMITY_PIXEL_THRESHOLD = 30.0   # max gap (px) between box edges to count as "touching/overlapping"
PROXIMITY_FRAMES_REQUIRED = 2      # consecutive qualifying frames before this signal is trusted

# --- Visual confirmation (crop-and-confirm) ---
# The vision model is still run - it's what tells "jeepney loading a
# passenger" / "two cars passing close together" apart from "jeepney
# just got hit" / "actual crash", which look identical to motion alone
# - but severity ITSELF is decided by the motion classifier below
# (classify_motion_severity), not by which of the model's three
# classes fires. The vision model's confidence is still logged
# (confirmation_confidence) and its class is still mapped to a tier
# via VISION_CLASS_TO_TIER purely so it can be compared against the
# motion tier and used to raise it if the vision model spotted
# something worse - see merge_severity().
ACCIDENT_SEVERITY_ORDER = ["severe", "moderate", "accident"]  # vision-model class priority, highest first
SEVERITY_LABELS = {
    "critical": "CRITICAL ACCIDENT",
    "severe": "SEVERE ACCIDENT",
    "moderate": "MODERATE ACCIDENT",
    "mild": "MILD ACCIDENT",
}
# BGR display colors per tier, used while a motion-classified event is
# still awaiting vision corroboration (vision-confirmed events always
# render magenta regardless of tier - see the display logic below).
TIER_DISPLAY_COLORS = {
    "critical": (0, 0, 255),     # red
    "severe": (0, 60, 255),      # red-orange
    "moderate": (0, 165, 255),   # orange
    "mild": (0, 220, 255),       # yellow-orange
}
# Maps our internal severity tier to the `severity` enum values in the
# database (schema.sql) - kept as a separate dict from SEVERITY_LABELS
# so a change to one doesn't silently break the other.
SEVERITY_DB_VALUES = {
    "critical": "CRITICAL",
    "severe": "SEVERE",
    "moderate": "MODERATE",
    "mild": "MILD",
}
CROP_PAD_RATIO = 1.0               # extra room around the box to include in the crop
CONFIRMATION_CHECK_COOLDOWN = 3.0  # seconds between re-checking the same ongoing event

# --- Event GIF ---
# In addition to the still crop, each new accident also gets a short GIF
# of the footage around the trigger - PRE_EVENT_SECONDS before it and
# POST_EVENT_SECONDS after it - so a reviewer can see what happened
# instead of one frozen frame. The last PRE_EVENT_SECONDS of video are
# kept in memory as compressed JPEGs (sampled at GIF_FPS); when an event
# triggers, that footage is snapshotted, the job keeps collecting frames
# for POST_EVENT_SECONDS more, and then the GIF is cropped, encoded, and
# uploaded in a background thread so the video keeps running.
GIF_ENABLED = True
PRE_EVENT_SECONDS = 7.0            # how much footage before the trigger to include
POST_EVENT_SECONDS = 5.0           # how much footage after the trigger to include (0 = none)
GIF_FPS = 10                       # frames per second sampled into the GIF (lower = smaller file)
GIF_MAX_WIDTH = 480                # GIFs wider than this get scaled down (lower = smaller file)
GIF_CROP_PAD_RATIO = 2.0           # padding around the crash area - bigger than CROP_PAD_RATIO so approaching vehicles show up; None = whole frame
GIF_MEDIA_TYPE = "IMAGE"           # `media_type` stored in accident_media (the dashboard prefers "IMAGE" over "CROP")

# --- Motion-based severity classification ---
# Severity is decided primarily from HOW the vehicle(s) moved, not
# from a single cropped frame the vision model happens to see -
# motion is available the instant an event triggers (no waiting on a
# confirmation cooldown), and two signals in particular correlate
# strongly with real-world crash severity:
#
#   - pre-crash speed: how fast the vehicle was going right before it
#     suddenly stopped (last_sudden_stop_speed). A 70 km/h -> 0 drop
#     is a fundamentally more violent event than a 10 km/h -> 0 one,
#     independent of how either happens to look in a still frame.
#   - box overlap (IoU): how much the two vehicles' boxes overlapped
#     at their most overlapping moment. Two boxes barely touching is
#     consistent with a light bump; boxes overlapping 50%+ means the
#     vehicles occupied nearly the same space, which only really
#     happens in a hard, deforming impact.
#
# Each tier below lists the MINIMUM pre-stop speed OR minimum peak IoU
# that qualifies for it - meeting either one is enough (whichever
# gives the higher tier wins). Checked highest-tier-first.
MOTION_SEVERITY_TIERS = [
    # (tier, min_pre_stop_speed_kmh, min_peak_iou)
    ("critical", 80.0, 0.65),
    ("severe",   55.0, 0.45),
    ("moderate", 35.0, 0.25),
    ("mild",      0.0, 0.00),   # catch-all floor: any real trigger that doesn't clear the above
]
TIER_RANK = {"mild": 1, "moderate": 2, "severe": 3, "critical": 4}

# A vehicle that's been sitting stopped for a long time after a sudden
# stop is more likely a disabled/damaged vehicle than one that's about
# to pull away - a long stop nudges the tier up by one step (capped at
# "critical") on top of whatever the speed/IoU check already found.
LONG_STOPPED_BUMP_SECONDS = 5.0

# --- Post-impact "flew away" detection ---
# A vehicle that gets knocked well away from the contact point and is
# still moving fast afterward (rather than stopping where it was hit)
# was hit hard - that's a much stronger severity signal than the boxes
# just touching. At first contact between two vehicles each one's
# position, speed, and heading are snapshotted; for THROWN_WINDOW_SECONDS
# afterward a vehicle is flagged as "flew away" if it ends up displaced
# from the contact point, is still moving fast, AND either changed
# direction sharply or sped up. A car that hits and stops, or hits and
# keeps going straight at the same speed, is NOT flagged.
THROWN_WINDOW_SECONDS = 2.0            # how long after first contact to watch for it
THROWN_MIN_DISPLACEMENT_M = 4.0        # must end up at least this far from the contact point
THROWN_MIN_POST_IMPACT_SPEED_KMH = 25.0
THROWN_MIN_DEFLECTION_DEG = 45.0       # direction change vs. pre-contact heading...
THROWN_MIN_SPEED_GAIN_KMH = 20.0       # ...OR speed increase vs. pre-contact speed
THROWN_MIN_HEADING_MOVE_M = 1.0        # below this, pre-contact heading is "unknown" (was ~stationary)
FLEW_AWAY_MIN_TIER = "severe"          # flying away bumps the tier one step, but never lands below this

# --- Severity locking ---
# The tier written to the database when an event first triggers (from
# motion) is final. Later re-checks still run the vision model and can
# record its confidence, but they can no longer change the severity -
# so a row that was queued as MODERATE stays MODERATE instead of being
# upgraded a moment later (which made the dashboard disagree with the
# alert that fired at insert time). Set to False to restore the old
# behavior where vision / long-stopped / flew-away can raise the tier.
LOCK_SEVERITY_AFTER_FIRST_CLASSIFICATION = True

# The vision model (if configured) is still consulted, but only as
# corroboration: MOTION_TO_VISION_RANK maps its raw class onto our
# tier scale so it can be compared against the motion-derived tier,
# and the HIGHER of the two always wins (see merge_severity below).
# This means a bad/under-confident vision read can never silently
# downgrade a real motion-confirmed crash, but a vision model that
# spots something worse than the motion signals alone suggested can
# still raise it.
VISION_CLASS_TO_TIER = {
    "severe": "severe",
    "moderate": "moderate",
    "accident": "mild",
}

ALERT_COOLDOWN_SECONDS = 5.0

# --- Debug snapshots (local disk only, NOT written to the database) ---
# Saves a frame every time MOTION or OVERLAP ALONE says "LIKELY
# ACCIDENT", even if the vision model never confirms it. Useful while
# tuning thresholds - lets you see what's triggering the logic without
# waiting for (or requiring) full accident-model confirmation, and
# without cluttering the real `accidents` table with unconfirmed
# guesses.
DEBUG_SAVE_TRIGGER_SNAPSHOTS = True
DEBUG_SNAPSHOT_DIR = "debug_snapshots"
if DEBUG_SAVE_TRIGGER_SNAPSHOTS:
    os.makedirs(DEBUG_SNAPSHOT_DIR, exist_ok=True)


# ============================================================
# LOAD MODELS
# ============================================================

def load_model(path, fallback_path=FALLBACK_MODEL_PATH):
    """Load a YOLO model, falling back to a shared/general model if the
    dedicated weights file isn't present. Lets this script keep running
    against `best.pt` (e.g. a single combined vehicle+accident model)
    if best_vehicle.pt / best_accident.pt haven't been trained or
    dropped into the folder yet, instead of crashing on startup."""
    if os.path.exists(path):
        return YOLO(path)
    if fallback_path and os.path.exists(fallback_path):
        print(f"[warn] '{path}' not found - falling back to '{fallback_path}'.")
        return YOLO(fallback_path)
    raise FileNotFoundError(
        f"Neither '{path}' nor fallback '{fallback_path}' could be found. "
        f"Place the model file in this script's folder or update the path."
    )


vehicle_model = load_model(VEHICLE_MODEL_PATH)
accident_model = load_model(ACCIDENT_MODEL_PATH)

print("Vehicle classes:")
for class_id, class_name in vehicle_model.names.items():
    print(f"  {class_id}: {class_name}")
print("Accident classes:")
for class_id, class_name in accident_model.names.items():
    print(f"  {class_id}: {class_name}")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

print("\nStarting vehicle detection & speed estimation...")
print("Press ESC to stop.")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0
delay = max(1, int(1000 / fps))


# ============================================================
# TRACK STATE
# ============================================================

track_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))                # (x, y, t, local_scale)
raw_positions = defaultdict(lambda: deque(maxlen=POSITION_SMOOTHING_FRAMES))  # for smoothing
speed_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
stopped_counter = defaultdict(int)
moving_counter = defaultdict(int)
stopped_since = defaultdict(lambda: None)
last_sudden_stop_time = defaultdict(lambda: -1e9)
last_sudden_stop_speed = defaultdict(float)  # how fast it was going right before the sudden stop
last_alert_time = defaultdict(lambda: -1e9)
track_age = defaultdict(int)
previous_status = defaultdict(lambda: None)

last_confirmation_check_time = defaultdict(lambda: -1e9)
confirmed_severity = defaultdict(lambda: None)   # current severity tier for this track's event (motion-classified immediately, possibly raised by vision later)
vision_confirmed_track = defaultdict(bool)       # has the vision model ever returned a result for this track's event (display only - doesn't affect the stored severity, which is motion-first)
last_debug_snapshot_time = defaultdict(lambda: -1e9)

# One database `accidents` row per ongoing event, not per frame.
# Reset to None once the track leaves the stopped/accident state, so
# the *next* event for the same track_id gets its own new row.
db_accident_id = defaultdict(lambda: None)

# --- Cross-track duplicate-crash guard ---
# Once a track_id has been visually CONFIRMED as part of an accident,
# it's locked out of starting any OTHER, brand-new accident event with
# a different vehicle. Without this, a car that has already been
# flagged as crashed - now sitting stopped/deformed in the road - would
# keep re-triggering "LIKELY ACCIDENT" (motion), box-overlap, or
# proximity+sudden-stop against every other vehicle that subsequently
# drives near it, spamming the `accidents` table with duplicate rows
# for what is really one single real-world crash. A track already
# mid-event (i.e. it already has a row) is NOT blocked from
# continuing/upgrading that same row - only from starting a *new* one.
involved_tracks = set()

# --- Post-impact "flew away" state ---
# impact_record: track_id -> snapshot (position, speed, heading) taken
# at first contact with another vehicle. thrown_vehicle: track_id ->
# True once that vehicle has been flagged as having flown away.
impact_record = {}
thrown_vehicle = defaultdict(bool)

# --- Event GIF buffer ---
# frame_buffer: (video_time, jpeg_bytes) for the last PRE_EVENT_SECONDS
# of footage. active_gif_jobs: GIFs whose trigger has happened but that
# are still collecting their POST_EVENT_SECONDS of footage.
frame_buffer = deque()
last_buffered_time = -1e9
active_gif_jobs = []
gif_threads = []

# --- Box-overlap tracking (IoU signal) ---
# Keyed by frozenset({track_id_a, track_id_b}) so the pair is
# order-independent. overlap_counter accumulates while the pair is
# overlapping and decays (not resets) on a frame where they briefly
# aren't, to absorb one-frame detector jitter without losing progress.
overlap_counter = defaultdict(int)

# --- Proximity + sudden-stop tracking (edge-distance signal) ---
# Same frozenset-pair keying as overlap_counter, but a SEPARATE
# counter: this signal only needs PROXIMITY_FRAMES_REQUIRED (a couple
# of frames) to trust, so it resets immediately rather than decaying -
# there's little accumulated evidence to protect, and a crisp reset
# keeps it reacting fast to a genuine hard-brake-into-another-vehicle
# moment instead of lingering from stale, unrelated near-misses.
proximity_counter = defaultdict(int)

# Highest IoU seen so far during THIS pair's ongoing event - used as
# the "how hard did they hit" signal for classify_motion_severity().
# Reset alongside the other per-pair state once the pair stops
# qualifying under both signals (see the counters' cleanup below).
overlap_peak_iou = defaultdict(float)

last_overlap_confirmation_check_time = defaultdict(lambda: -1e9)
confirmed_overlap_severity = defaultdict(lambda: None)
vision_confirmed_overlap = defaultdict(bool)
last_overlap_debug_snapshot_time = defaultdict(lambda: -1e9)
last_overlap_alert_time = defaultdict(lambda: -1e9)
db_accident_id_for_pair = defaultdict(lambda: None)

# --- ID-switch stitching state ---
# canonical_id maps every RAW track_id the tracker has ever emitted to
# the "real" (canonical) track_id we treat it as. For a track_id we've
# never seen before this is usually itself - unless it looks like the
# continuation of a track that just vanished nearby, in which case it
# maps to that older track's canonical id instead.
canonical_id = {}
# Last frame index / centroid / class we saw for each CANONICAL
# track_id (not raw id) - used to decide whether a new raw id is
# "close enough, soon enough, same kind of vehicle" to be the same
# vehicle reappearing after a tracker hiccup.
last_seen_frame_by_canonical = {}
last_seen_pos_by_canonical = {}
last_seen_class_by_canonical = {}


def resolve_track_id(raw_id, cx, cy, class_name, frame_idx, claimed_this_frame):
    """Map a raw tracker id to a stable canonical id, stitching across
    tracker-induced ID switches (e.g. the box churn during a crash).

    First time we see `raw_id`, check whether some other canonical
    track went missing within STITCH_MAX_FRAMES_GAP frames, within
    STITCH_MAX_DIST_PX of this new box's centroid, is the same vehicle
    class, and - critically - is NOT already accounted for elsewhere
    in THIS SAME FRAME (either because its raw id is still being
    tracked live, or because another new raw id already claimed it a
    moment ago in this same frame's resolution pass). That last check
    is what prevents two different physical vehicles from ever ending
    up sharing one canonical id within a single frame: a canonical
    track that's still actually on screen right now can never be
    "the same vehicle" as some other box also on screen right now.
    If a real, unclaimed candidate is found, `raw_id` is treated as
    that vehicle continuing under a new tracker id. Otherwise `raw_id`
    becomes its own canonical id.
    """
    if raw_id in canonical_id:
        return canonical_id[raw_id]

    DIAGNOSTIC_RADIUS_PX = STITCH_MAX_DIST_PX * 3  # only log candidates plausibly relevant, not every unrelated car in the scene
    best_match = None
    best_dist = STITCH_MAX_DIST_PX
    near_misses = []  # for STITCH_DEBUG: nearby-ish candidates that were rejected, and why
    for old_canonical, (ox, oy) in last_seen_pos_by_canonical.items():
        dist = math.hypot(cx - ox, cy - oy)
        gap = frame_idx - last_seen_frame_by_canonical[old_canonical]
        worth_logging = STITCH_DEBUG and dist <= DIAGNOSTIC_RADIUS_PX

        if old_canonical in claimed_this_frame:
            if worth_logging:
                near_misses.append((old_canonical, f"already active this frame (dist={dist:.0f}px)"))
            continue  # already active (or already stitched to) elsewhere in this exact frame
        old_class = last_seen_class_by_canonical.get(old_canonical)
        if old_class != class_name:
            if worth_logging:
                near_misses.append((old_canonical, f"class mismatch ({old_class}!={class_name}, dist={dist:.0f}px)"))
            continue  # don't stitch a car onto a motorcycle's old id, etc.
        if gap <= 0:
            if worth_logging:
                near_misses.append((old_canonical, f"still active this frame (dist={dist:.0f}px)"))
            continue
        if gap > STITCH_MAX_FRAMES_GAP:
            if worth_logging:
                near_misses.append((old_canonical, f"gap too big ({gap}f > {STITCH_MAX_FRAMES_GAP}f, dist={dist:.0f}px)"))
            continue
        if dist > STITCH_MAX_DIST_PX:
            if worth_logging:
                near_misses.append((old_canonical, f"too far ({dist:.0f}px > {STITCH_MAX_DIST_PX}px, gap={gap}f)"))
            continue
        if dist <= best_dist:
            best_dist = dist
            best_match = old_canonical

    canonical_id[raw_id] = best_match if best_match is not None else raw_id
    if best_match is not None:
        claimed_this_frame.add(best_match)  # so no OTHER new raw id this frame can also claim it
        if DEBUG_PRINT_STATUS_CHANGES:
            print(f"  [debug] stitched raw track {raw_id} -> canonical track {best_match} "
                  f"(dist={best_dist:.1f}px)")
    elif STITCH_DEBUG and near_misses:
        # This raw id became its OWN new canonical id despite having
        # at least one plausibly-nearby rejected candidate - print why
        # each was rejected, so a real missed stitch (gap or distance
        # threshold just slightly too tight) is visible in the log
        # instead of failing silently.
        print(f"  [stitch-debug] t={frame_idx/max(fps,1):6.2f}s NEW canonical track {raw_id} "
              f"({class_name}) at ({cx:.0f},{cy:.0f}) - rejected nearby candidates: "
              + "; ".join(f"{cid}:{reason}" for cid, reason in near_misses))
    return canonical_id[raw_id]


def local_pixels_per_meter(class_name, box_width_px):
    real_width = VEHICLE_REAL_WIDTH_M.get(class_name)
    if not real_width or box_width_px <= 0:
        return FALLBACK_PIXELS_PER_METER
    return box_width_px / real_width


def compute_speed_kmh(history):
    """Estimate speed (km/h) from a deque of (x, y, t, local_scale)
    samples, using the oldest sample within HISTORY_SECONDS and the
    newest sample."""
    if len(history) < 2:
        return 0.0
    newest = history[-1]
    oldest = history[0]
    for point in history:
        if newest[2] - point[2] <= HISTORY_SECONDS:
            oldest = point
            break
    dt = newest[2] - oldest[2]
    if dt <= 0:
        return 0.0

    dx = newest[0] - oldest[0]
    dy = newest[1] - oldest[1]
    pixel_dist = math.hypot(dx, dy)

    meters = pixel_dist / newest[3]
    mps = meters / dt
    return mps * 3.6


def compute_iou(box_a, box_b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_edge_distance(box_a, box_b):
    """Gap (px) between the nearest edges of two boxes. 0 if the boxes
    touch or overlap; grows as they move apart."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    dy = max(0.0, max(ay1, by1) - min(ay2, by2))
    return math.hypot(dx, dy)


def get_status(track_id, speed_kmh, now):
    """Return 'MOVING', 'STOPPED', 'SUDDEN STOP', or 'LIKELY ACCIDENT'."""
    status = "MOVING"

    if speed_kmh < STOPPED_SPEED_KMH:
        stopped_counter[track_id] += 1
        moving_counter[track_id] = 0
        if stopped_since[track_id] is None:
            stopped_since[track_id] = now
    else:
        moving_counter[track_id] += 1
        # Only give up on an accumulating stop after a SUSTAINED run of
        # above-threshold frames, not a single noisy one - a crashed
        # vehicle rocking slightly, dust/debris, or a brief partial
        # occlusion can all cause a one-frame speed blip even though
        # the vehicle never actually moved.
        if moving_counter[track_id] >= MOVING_FRAMES_TO_RESET:
            stopped_counter[track_id] = 0
            stopped_since[track_id] = None
            # The vehicle is confirmed moving again - the event (if any)
            # for this track has ended. Next time it stops, that's a
            # new event and should get a new database row, and the
            # pre-stop speed from the OLD event shouldn't carry over.
            db_accident_id[track_id] = None
            confirmed_severity[track_id] = None
            vision_confirmed_track[track_id] = False
            last_sudden_stop_speed[track_id] = 0.0

    if stopped_counter[track_id] >= STOPPED_FRAMES_REQUIRED:
        status = "STOPPED"

    hist = speed_history[track_id]
    hist.append((speed_kmh, now))
    max_recent_speed = 0.0
    for spd, t in hist:
        if now - t <= SUDDEN_STOP_WINDOW:
            max_recent_speed = max(max_recent_speed, spd)

    drop = max_recent_speed - speed_kmh
    if drop >= SUDDEN_STOP_DROP_KMH and speed_kmh < max_recent_speed:
        status = "SUDDEN STOP"
        last_sudden_stop_time[track_id] = now
        last_sudden_stop_speed[track_id] = max_recent_speed

    # A sudden stop from highway/road speed is much stronger evidence
    # of a real crash than one from a slow roll - require far less
    # "time spent stopped" afterward before escalating to accident.
    required_stopped_duration = (
        STOPPED_DURATION_FOR_ACCIDENT_HIGH_SPEED
        if last_sudden_stop_speed[track_id] >= HIGH_SPEED_SUDDEN_STOP_KMH
        else STOPPED_DURATION_FOR_ACCIDENT
    )

    time_since_sudden_stop = now - last_sudden_stop_time[track_id]
    if (
        status == "STOPPED"
        and time_since_sudden_stop <= SUDDEN_STOP_TO_ACCIDENT_WINDOW
        and stopped_since[track_id] is not None
        and (now - stopped_since[track_id]) >= required_stopped_duration
    ):
        status = "LIKELY ACCIDENT"

    return status


def check_accident_visual(crop):
    """Run the dedicated accident model on an already-cropped image.
    Returns (tier, confidence) for the highest-priority class found,
    mapped through VISION_CLASS_TO_TIER, or (None, 0.0) if nothing
    cleared ACCIDENT_CONFIDENCE. This is corroboration only now -
    severity itself comes from classify_motion_severity() below; this
    result is merged in afterward and can only raise the tier, never
    lower it. See merge_severity()."""
    if crop is None or crop.size == 0:
        return None, 0.0

    results = accident_model.predict(crop, conf=ACCIDENT_CONFIDENCE, verbose=False)
    found = {}
    for box in results[0].boxes:
        name = accident_model.names[int(box.cls[0])].lower()
        conf = float(box.conf[0])
        if name in ACCIDENT_SEVERITY_ORDER:
            found[name] = max(found.get(name, 0.0), conf)

    for model_class in ACCIDENT_SEVERITY_ORDER:
        if model_class in found:
            return VISION_CLASS_TO_TIER[model_class], found[model_class]
    return None, 0.0


# ------------------------------------------------------------
# Post-impact "flew away" helpers
# ------------------------------------------------------------

def angle_diff_deg(a, b):
    """Smallest absolute difference between two angles in degrees (0-180)."""
    d = abs(a - b) % 360
    return 360 - d if d > 180 else d


def compute_heading_deg(history):
    """Direction of travel (image coords) over the history window, or
    None if the vehicle barely moved."""
    if len(history) < 2:
        return None
    x0, y0, _, _ = history[0]
    x1, y1, _, scale = history[-1]
    if math.hypot(x1 - x0, y1 - y0) / scale < THROWN_MIN_HEADING_MOVE_M:
        return None
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def start_impact_record(track_id, now):
    """Snapshot this track's position/speed/heading at first contact."""
    hist = track_history[track_id]
    if len(hist) < 2:
        return
    x, y, _, _ = hist[-1]
    impact_record[track_id] = {
        "t0": now, "x": x, "y": y,
        "pre_speed": compute_speed_kmh(hist),
        "pre_heading": compute_heading_deg(hist),
    }
    thrown_vehicle[track_id] = False


def note_contact(track_id, now):
    """Called on every frame a pair qualifies as in contact. Only starts
    a fresh record if there's none, or the old one expired on a track
    that isn't already part of a logged crash."""
    rec = impact_record.get(track_id)
    if rec is not None and (track_id in involved_tracks or now - rec["t0"] <= THROWN_WINDOW_SECONDS):
        return
    start_impact_record(track_id, now)


def update_thrown_flag(track_id, now, x, y, scale, speed_kmh):
    """Within THROWN_WINDOW_SECONDS of first contact, flag the vehicle
    as having flown away if it's displaced from the contact point, still
    moving fast, and either deflected sharply or sped up."""
    rec = impact_record.get(track_id)
    if rec is None or thrown_vehicle[track_id]:
        return
    if now - rec["t0"] > THROWN_WINDOW_SECONDS:
        return

    dx, dy = x - rec["x"], y - rec["y"]
    displacement_m = math.hypot(dx, dy) / scale
    if displacement_m < THROWN_MIN_DISPLACEMENT_M or speed_kmh < THROWN_MIN_POST_IMPACT_SPEED_KMH:
        return

    speed_gain = speed_kmh - rec["pre_speed"]
    if rec["pre_heading"] is None:
        deflection = 180.0   # was ~stationary before contact, so any motion is "new"
    else:
        deflection = angle_diff_deg(math.degrees(math.atan2(dy, dx)), rec["pre_heading"])

    if deflection >= THROWN_MIN_DEFLECTION_DEG or speed_gain >= THROWN_MIN_SPEED_GAIN_KMH:
        thrown_vehicle[track_id] = True
        print(f"  [debug] t={now:6.2f}s track {track_id} FLEW AWAY "
              f"(moved {displacement_m:.1f} m, {speed_kmh:.0f} km/h, "
              f"deflection={deflection:.0f} deg, speed gain={speed_gain:.0f} km/h)")


def bump_tier(tier, steps=1):
    """Move a tier up by `steps`, capped at the highest tier."""
    tiers_by_rank = sorted(TIER_RANK, key=TIER_RANK.get)
    return tiers_by_rank[min(tiers_by_rank.index(tier) + steps, len(tiers_by_rank) - 1)]


def classify_motion_severity(pre_stop_speed_kmh=0.0, peak_iou=0.0, stopped_duration_s=0.0,
                             flew_away=False):
    """Decide a severity tier straight from how the vehicle(s) moved -
    no vision model involved. Checks MOTION_SEVERITY_TIERS
    highest-tier-first and returns the first one where EITHER the
    pre-stop speed or the peak box-overlap (IoU) clears that tier's
    minimum; the "mild" entry has thresholds of 0 so it always matches
    as the floor for any real trigger. A long post-impact stopped
    duration then nudges the result up by one tier on top of that (a
    vehicle still sitting there disabled minutes later is worse than
    one that grazed another and kept going, even at the same impact
    speed). If a vehicle flew away after impact, the tier is bumped one
    more step and floored at FLEW_AWAY_MIN_TIER."""
    tier = "mild"
    for candidate_tier, min_speed, min_iou in MOTION_SEVERITY_TIERS:
        if pre_stop_speed_kmh >= min_speed or peak_iou >= min_iou:
            tier = candidate_tier
            break

    if stopped_duration_s >= LONG_STOPPED_BUMP_SECONDS:
        tier = bump_tier(tier)

    if flew_away:
        tier = bump_tier(tier)
        if TIER_RANK[tier] < TIER_RANK[FLEW_AWAY_MIN_TIER]:
            tier = FLEW_AWAY_MIN_TIER

    return tier


def merge_severity(motion_tier, vision_tier):
    """Combine the motion-derived tier with the (optional) vision
    model's corroborating tier by taking the HIGHER of the two -
    vision can raise a motion-based read if it spots something worse,
    but a low-confidence or absent vision result can never silently
    downgrade a real motion-confirmed event."""
    if vision_tier is None:
        return motion_tier
    if TIER_RANK[vision_tier] > TIER_RANK[motion_tier]:
        return vision_tier
    return motion_tier


def padded_rect(frame_shape, x1, y1, x2, y2, pad_ratio):
    """(x1, y1, x2, y2) plus `pad_ratio` extra room on each side,
    clamped to the frame bounds, as ints."""
    h, w = frame_shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = box_w * pad_ratio, box_h * pad_ratio

    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))
    return cx1, cy1, cx2, cy2


def padded_crop(frame, x1, y1, x2, y2, pad_ratio):
    """Crop `frame` to (x1, y1, x2, y2) plus `pad_ratio` extra room on
    each side, clamped to the frame bounds."""
    cx1, cy1, cx2, cy2 = padded_rect(frame.shape, x1, y1, x2, y2, pad_ratio)
    return frame[cy1:cy2, cx1:cx2]


def union_box(box_a, box_b):
    """Smallest box containing both boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)


def video_time_to_walltime(video_seconds):
    """Convert frame_idx/fps video-relative seconds into a real
    timestamp, anchored to when this run started."""
    return VIDEO_START_WALLTIME + timedelta(seconds=video_seconds)


def save_or_update_accident(accident_id, track_ids, class_names, severity_db_value,
                             detection_conf, confirmation_conf, now, crop_image, media_type="CROP"):
    """Insert a new `accidents` row (when `accident_id` is None) or
    update an existing one in place (used to upgrade a row that was
    logged as 'UNCLASSIFIED'/pending-review the moment motion or
    overlap first triggered, once the vision model later actually
    confirms a real severity - so the SAME row gets upgraded instead
    of a second row being created for the same event).

    Either way, uploads `crop_image` as a new `accident_media` row and
    refreshes `captured_image_url`. `track_ids`/`class_names` can be a
    single-item list (motion trigger) or two items (overlap trigger,
    one per vehicle in the pair). Returns the accident id (new or the
    one passed in), or None if Supabase isn't configured or the
    initial insert fails."""
    if not SUPABASE_ENABLED:
        return None

    track_id_label = "_".join(str(t) for t in track_ids)

    try:
        if accident_id is None:
            insert_res = supabase.table("accidents").insert({
                "timestamp": video_time_to_walltime(now).isoformat(),
                "location": CAMERA_LOCATION,
                "camera_id": CAMERA_ID,
                "detection_confidence": detection_conf,
                "confirmation_confidence": confirmation_conf,
                "severity": severity_db_value,
                "vehicle_types": list(dict.fromkeys(class_names)),  # de-duped, order preserved
                "track_id": track_id_label,
                "ai_source_id": "vehicle_accident_detection.py",
            }).execute()
            accident_id = insert_res.data[0]["id"]
            print(f"[db] queued accident {accident_id} for review "
                  f"(track {track_id_label}, severity={severity_db_value})")
        else:
            supabase.table("accidents").update({
                "severity": severity_db_value,
                "confirmation_confidence": confirmation_conf,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", accident_id).execute()
            print(f"[db] upgraded accident {accident_id} "
                  f"(track {track_id_label}) to severity={severity_db_value}")

        if crop_image is not None and crop_image.size > 0:
            ok, buf = cv2.imencode(".jpg", crop_image)
            if ok:
                storage_path = f"{accident_id}/{media_type.lower()}_{track_id_label}_{now:.2f}.jpg"
                supabase.storage.from_(STORAGE_BUCKET).upload(
                    storage_path,
                    buf.tobytes(),
                    file_options={"content-type": "image/jpeg"},
                )
                supabase.table("accident_media").insert({
                    "accident_id": accident_id,
                    "media_type": media_type,
                    "storage_bucket": STORAGE_BUCKET,
                    "storage_path": storage_path,
                }).execute()
                # Denormalized convenience field on the accident row
                # itself, for dashboards that just want a quick thumbnail
                # without a second query into accident_media.
                supabase.table("accidents").update({
                    "captured_image_url": storage_path
                }).eq("id", accident_id).execute()

        return accident_id

    except Exception as e:
        print(f"[db] failed to write/update accident for track {track_id_label}: {e}")
        return accident_id


# ------------------------------------------------------------
# Pre-event GIF
# ------------------------------------------------------------

def buffer_frame_for_gif(frame, now):
    """Keep the last PRE_EVENT_SECONDS of footage as JPEG bytes, sampled
    at GIF_FPS, and also hand each sampled frame to any GIF job that is
    still collecting its post-event footage. JPEG-compressing (instead
    of holding raw frames) keeps memory small while still letting the
    GIF be cropped at full resolution later."""
    global last_buffered_time
    if not (GIF_ENABLED and SUPABASE_ENABLED):
        return
    # Small tolerance so float rounding (e.g. 0.3 - 0.2 = 0.0999...) doesn't
    # skip a frame that's really on the GIF_FPS grid.
    if (now - last_buffered_time) < (1.0 / GIF_FPS) - 1e-3:
        return
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return
    jpg = buf.tobytes()
    frame_buffer.append((now, jpg))
    last_buffered_time = now
    while frame_buffer and (now - frame_buffer[0][0]) > PRE_EVENT_SECONDS:
        frame_buffer.popleft()
    for job in active_gif_jobs:
        if now <= job["end_time"]:
            job["frames"].append((now, jpg))


def flush_finished_gif_jobs(now=None, force=False):
    """Hand every GIF job whose post-event window has elapsed (or all of
    them, when `force` is set because the video ended) to a background
    thread that builds and uploads the GIF."""
    for job in list(active_gif_jobs):
        if force or (now is not None and now >= job["end_time"]):
            active_gif_jobs.remove(job)
            t = threading.Thread(
                target=_build_and_upload_gif,
                args=(job["accident_id"], job["label"], job["frames"],
                      job["rect"], job["trigger_time"]),
                daemon=True,
            )
            t.start()
            gif_threads.append(t)


def _build_and_upload_gif(accident_id, label, snapshot, rect, now):
    """Runs in a background thread: crop the buffered frames to `rect`,
    encode them as a GIF, upload it, and add an `accident_media` row."""
    try:
        cx1, cy1, cx2, cy2 = rect
        pil_frames = []
        for _, jpg in snapshot:
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            h, w = crop.shape[:2]
            if w > GIF_MAX_WIDTH:
                scale = GIF_MAX_WIDTH / w
                crop = cv2.resize(crop, (GIF_MAX_WIDTH, max(1, int(h * scale))),
                                  interpolation=cv2.INTER_AREA)
            pil_frames.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

        if len(pil_frames) < 2:
            print(f"[gif] not enough buffered frames for accident {accident_id} - skipped")
            return

        # Play back at the real elapsed speed even if the video's fps is
        # lower than GIF_FPS.
        span = snapshot[-1][0] - snapshot[0][0]
        if span > 0:
            duration_ms = max(20, int(1000 * span / (len(pil_frames) - 1)))
        else:
            duration_ms = int(1000 / GIF_FPS)

        buf = io.BytesIO()
        pil_frames[0].save(
            buf, format="GIF", save_all=True, append_images=pil_frames[1:],
            duration=duration_ms, loop=0,
        )
        data = buf.getvalue()

        storage_path = f"{accident_id}/gif_{label}_{now:.2f}.gif"
        supabase.storage.from_(STORAGE_BUCKET).upload(
            storage_path, data, file_options={"content-type": "image/gif"},
        )
        supabase.table("accident_media").insert({
            "accident_id": accident_id,
            "media_type": GIF_MEDIA_TYPE,
            "storage_bucket": STORAGE_BUCKET,
            "storage_path": storage_path,
        }).execute()
        print(f"[gif] uploaded {len(pil_frames)}-frame GIF ({len(data) / 1024:.0f} KB) "
              f"for accident {accident_id}")
    except Exception as e:
        print(f"[gif] failed for accident {accident_id}: {e}")


def queue_event_gif(accident_id, label, frame_shape, x1, y1, x2, y2, now):
    """Start a GIF job for this accident: snapshot the pre-event footage
    now, keep collecting frames for POST_EVENT_SECONDS, then (see
    flush_finished_gif_jobs) build and upload it. `x1..y2` is the crash
    area (single box or union of a pair); the same region is cropped out
    of every frame."""
    if not (GIF_ENABLED and SUPABASE_ENABLED) or accident_id is None:
        return
    if GIF_CROP_PAD_RATIO is None:
        h, w = frame_shape[:2]
        rect = (0, 0, w, h)
    else:
        rect = padded_rect(frame_shape, x1, y1, x2, y2, GIF_CROP_PAD_RATIO)
    active_gif_jobs.append({
        "accident_id": accident_id,
        "label": label,
        "rect": rect,
        "frames": list(frame_buffer),
        "trigger_time": now,
        "end_time": now + POST_EVENT_SECONDS,
    })


# ============================================================
# DETECTION + TRACKING LOOP
# ============================================================

frame_idx = 0

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # Video time, not wall-clock - keeps speed math correct even if
    # inference runs slower/faster than real-time on a recorded file.
    now = frame_idx / fps

    buffer_frame_for_gif(frame, now)
    flush_finished_gif_jobs(now)

    results = vehicle_model.track(
        frame,
        conf=CONFIDENCE,
        iou=NMS_IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    annotated_frame = frame.copy()

    boxes = results[0].boxes
    box_by_track_id = {}          # canonical track_id -> (x1, y1, x2, y2), this frame only
    class_name_by_track_id = {}
    status_by_track_id = {}       # canonical track_id -> status string, this frame only
    track_age_ok = {}             # canonical track_id -> bool, whether it's old enough to trust
    canonical_ids_seen_this_frame = set()

    if boxes is not None and boxes.id is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        raw_ids = boxes.id.cpu().numpy().astype(int)
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        # Canonical ids that are ALREADY active this frame under a raw
        # id the tracker has kept stable (i.e. raw_id already appears
        # in canonical_id) must never also be handed to some OTHER,
        # brand-new raw id in this same frame as a "stitch" - that
        # canonical vehicle is provably still on screen right now under
        # its own id, so it can't simultaneously be a different box.
        # Pre-computing this before resolving any new ids removes the
        # dependency on box processing order that caused two different
        # cars to both display "ID 2" in the same frame.
        claimed_this_frame = set()
        for raw_id in raw_ids:
            existing_canonical = canonical_id.get(int(raw_id))
            if existing_canonical is not None:
                claimed_this_frame.add(existing_canonical)

        # ------------------------------------------------------
        # PASS 1: per-track motion status, drawing, book-keeping.
        # (Overlap pairs are handled in PASS 2, after every box for
        # this frame has already been drawn onto annotated_frame -
        # that way the pair's crash image shows both boxes, however
        # far apart they are in the detection order.)
        # ------------------------------------------------------
        for box, raw_track_id, cls_id, conf in zip(xyxy, raw_ids, clss, confs):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Resolve the raw tracker id to a stable canonical id
            # BEFORE touching any per-track state, so a tracker-level
            # ID switch (e.g. right at the moment of a collision, when
            # the box shape deforms or gets briefly occluded) doesn't
            # make this vehicle look like a brand-new track with no
            # history/status/db-row continuity.
            box_width_px = x2 - x1
            class_name = vehicle_model.names.get(cls_id, str(cls_id))

            track_id = resolve_track_id(int(raw_track_id), cx, cy, class_name, frame_idx, claimed_this_frame)
            canonical_ids_seen_this_frame.add(track_id)

            box_by_track_id[track_id] = (x1, y1, x2, y2)
            class_name_by_track_id[track_id] = class_name

            track_age[track_id] += 1
            track_age_ok[track_id] = track_age[track_id] >= MIN_TRACK_AGE_FRAMES

            # Smooth the raw centroid over a few frames before it
            # turns into noisy "speed."
            raw_positions[track_id].append((cx, cy))
            smoothed_x = sum(p[0] for p in raw_positions[track_id]) / len(raw_positions[track_id])
            smoothed_y = sum(p[1] for p in raw_positions[track_id]) / len(raw_positions[track_id])

            scale = local_pixels_per_meter(class_name, box_width_px)
            track_history[track_id].append((smoothed_x, smoothed_y, now, scale))

            speed_kmh = compute_speed_kmh(track_history[track_id])

            # Post-impact "flew away" check (no-op unless this track has
            # an impact record from a recent contact - see PASS 2).
            update_thrown_flag(track_id, now, smoothed_x, smoothed_y, scale, speed_kmh)

            raw_status = get_status(track_id, speed_kmh, now)

            # Don't trust an alarming status from a track that's brand
            # new or just got a fresh ID after an occlusion/ID switch.
            status = "MOVING" if not track_age_ok[track_id] else raw_status
            status_by_track_id[track_id] = status

            if DEBUG_PRINT_STATUS_CHANGES and status != previous_status[track_id]:
                stopped_for = (now - stopped_since[track_id]) if stopped_since[track_id] is not None else 0.0
                pre_stop = last_sudden_stop_speed[track_id]
                high_flag = " [HIGH-SPEED STOP]" if pre_stop >= HIGH_SPEED_SUDDEN_STOP_KMH else ""
                print(f"  [debug] t={now:6.2f}s track {track_id} ({class_name}): "
                      f"{previous_status[track_id]} -> {status} "
                      f"(speed={speed_kmh:.1f} km/h, was={pre_stop:.1f} km/h, stopped_for={stopped_for:.2f}s){high_flag}")
                previous_status[track_id] = status

            # --- Log for human review immediately, upgrade if/when vision confirms ---
            severity = None
            crop_image = None
            if status == "LIKELY ACCIDENT":
                if DEBUG_SAVE_TRIGGER_SNAPSHOTS and (now - last_debug_snapshot_time[track_id]) >= CONFIRMATION_CHECK_COOLDOWN:
                    snap_path = os.path.join(
                        DEBUG_SNAPSHOT_DIR,
                        f"track{track_id}_{class_name}_t{now:.2f}s.jpg"
                    )
                    cv2.imwrite(snap_path, frame)
                    last_debug_snapshot_time[track_id] = now
                    print(f"  [debug] motion-only trigger snapshot saved: {snap_path}")

                # Write the row the moment motion alone says "likely
                # accident" - severity is classified straight from HOW
                # the vehicle moved (pre-crash speed, how long it's
                # been stopped), not left as a placeholder waiting on
                # the vision model - so a real tier shows up in the
                # dashboard immediately instead of several seconds later.
                #
                # Skip creating a brand-new row if this track is already
                # locked into a DIFFERENT, already-classified accident
                # (involved_tracks) - it's almost certainly the same
                # stopped/deformed vehicle re-triggering against a new
                # neighbor, not a second real crash.
                if db_accident_id[track_id] is None and track_id not in involved_tracks:
                    pending_crop = padded_crop(frame, x1, y1, x2, y2, CROP_PAD_RATIO)
                    motion_tier = classify_motion_severity(
                        pre_stop_speed_kmh=last_sudden_stop_speed[track_id],
                        stopped_duration_s=(
                            (now - stopped_since[track_id]) if stopped_since[track_id] is not None else 0.0
                        ),
                        flew_away=thrown_vehicle[track_id],
                    )
                    confirmed_severity[track_id] = motion_tier
                    involved_tracks.add(track_id)  # this is a real, motion-classified event now, not a placeholder
                    db_accident_id[track_id] = save_or_update_accident(
                        None, [track_id], [class_name], SEVERITY_DB_VALUES[motion_tier],
                        detection_conf=float(conf),
                        confirmation_conf=0.0,  # not yet corroborated by vision
                        now=now,
                        crop_image=pending_crop,
                    )
                    queue_event_gif(db_accident_id[track_id], str(track_id), frame.shape,
                                        x1, y1, x2, y2, now)

                if db_accident_id[track_id] is not None and (now - last_confirmation_check_time[track_id]) >= CONFIRMATION_CHECK_COOLDOWN:
                    crop_image = padded_crop(frame, x1, y1, x2, y2, CROP_PAD_RATIO)
                    vision_tier, vision_conf = check_accident_visual(crop_image)
                    # Re-run the motion classifier too, not just vision -
                    # pre-crash speed is fixed once the sudden stop
                    # happened, but stopped_duration keeps growing, so a
                    # track that's been sitting there a long time can
                    # still get bumped up later even if the initial
                    # trigger was milder.
                    motion_tier = classify_motion_severity(
                        pre_stop_speed_kmh=last_sudden_stop_speed[track_id],
                        stopped_duration_s=(
                            (now - stopped_since[track_id]) if stopped_since[track_id] is not None else 0.0
                        ),
                        flew_away=thrown_vehicle[track_id],
                    )
                    newly_vision_confirmed = vision_tier is not None and not vision_confirmed_track[track_id]
                    if vision_tier is not None:
                        vision_confirmed_track[track_id] = True
                    merged_tier = merge_severity(motion_tier, vision_tier)
                    if DEBUG_PRINT_STATUS_CHANGES and merged_tier != confirmed_severity[track_id]:
                        print(f"  [debug] t={now:6.2f}s track {track_id} severity re-check: "
                              f"stored={confirmed_severity[track_id]}, motion={motion_tier}, "
                              f"vision={vision_tier} (conf={vision_conf:.4f}), "
                              f"flew_away={thrown_vehicle[track_id]} -> "
                              f"{'kept (locked)' if LOCK_SEVERITY_AFTER_FIRST_CLASSIFICATION else 'updating to ' + merged_tier}")
                    if LOCK_SEVERITY_AFTER_FIRST_CLASSIFICATION:
                        # Tier stays as first written. Just record vision's
                        # confidence once (no tier change, no extra crop).
                        if newly_vision_confirmed:
                            save_or_update_accident(
                                db_accident_id[track_id], [track_id], [class_name],
                                SEVERITY_DB_VALUES[confirmed_severity[track_id]],
                                detection_conf=float(conf),
                                confirmation_conf=vision_conf,
                                now=now,
                                crop_image=None,
                            )
                    elif merged_tier != confirmed_severity[track_id]:
                        confirmed_severity[track_id] = merged_tier
                        save_or_update_accident(
                            db_accident_id[track_id], [track_id], [class_name],
                            SEVERITY_DB_VALUES[merged_tier],
                            detection_conf=float(conf),
                            confirmation_conf=vision_conf,
                            now=now,
                            crop_image=crop_image,
                        )
                    last_confirmation_check_time[track_id] = now
                severity = confirmed_severity[track_id]

            if severity is not None:
                color = TIER_DISPLAY_COLORS[severity]
                if vision_confirmed_track[track_id]:
                    display_status = f"CONFIRMED {SEVERITY_LABELS[severity]}"
                    color = (255, 0, 255)   # magenta - motion AND vision agree
                else:
                    # Motion alone has already classified this as a
                    # real event (not a placeholder) - vision just
                    # hasn't weighed in yet (or its checks haven't
                    # cleared ACCIDENT_CONFIDENCE), so the tier shown
                    # here is a genuine, if not yet vision-corroborated,
                    # severity - not a "pending review" placeholder.
                    display_status = f"{SEVERITY_LABELS[severity]} (motion)"
            elif status in ("STOPPED", "LIKELY ACCIDENT"):
                display_status = "STOPPED"
                color = (0, 165, 255)   # orange
            elif status == "SUDDEN STOP":
                display_status = "SUDDEN STOP"
                color = (0, 0, 255)     # red
            else:
                display_status = "MOVING"
                color = (0, 255, 0)     # green

            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label_1 = f"ID {track_id} {class_name} {conf:.2f}"
            label_2 = f"{speed_kmh:.1f} km/h - {display_status}"

            cv2.putText(annotated_frame, label_1, (int(x1), int(y1) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(annotated_frame, label_2, (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Only alert once BOTH motion and vision agree, debounced
            # per-track so one event doesn't spam the console.
            if severity is not None and (now - last_alert_time[track_id]) >= ALERT_COOLDOWN_SECONDS:
                print(f"[ALERT] Track {track_id} ({class_name}): CONFIRMED {SEVERITY_LABELS[severity]} "
                      f"- motion and vision both agree, now {speed_kmh:.1f} km/h")
                last_alert_time[track_id] = now

            # Record this canonical track's latest position/frame/class
            # for the stitcher to match against if it disappears later.
            last_seen_frame_by_canonical[track_id] = frame_idx
            last_seen_pos_by_canonical[track_id] = (cx, cy)
            last_seen_class_by_canonical[track_id] = class_name

        # ------------------------------------------------------
        # RESCUE PASS: the main detection pass (above, at CONFIDENCE)
        # can genuinely miss a vehicle - dark paint, harsh shadow, odd
        # post-impact angle - even when nothing overlapped it and NMS
        # never touched it. Rather than lowering CONFIDENCE globally
        # (which would add detection flicker/false positives for the
        # WHOLE video), only relax the threshold in the narrow case
        # where a missed vehicle would actually matter: right around
        # something that just registered SUDDEN STOP. Crop tightly
        # around that vehicle, re-run the detector there at
        # RESCUE_CONFIDENCE, and treat anything found that ISN'T
        # already one of this frame's real tracks as a likely hidden
        # crash partner. Rescued boxes never become their own
        # persistent track_id - they're a same-frame visual/DB signal
        # only, keyed to whichever real vehicle they were found next to.
        # ------------------------------------------------------
        for sudden_stop_id, sudden_stop_box in list(box_by_track_id.items()):
            if status_by_track_id.get(sudden_stop_id) != "SUDDEN STOP":
                continue
            if sudden_stop_id in involved_tracks:
                continue  # already confirmed part of a different crash - don't re-trigger

            sx1, sy1, sx2, sy2 = sudden_stop_box
            sw, sh = sx2 - sx1, sy2 - sy1
            rx1 = max(0, int(sx1 - sw * RESCUE_SEARCH_RADIUS_RATIO))
            ry1 = max(0, int(sy1 - sh * RESCUE_SEARCH_RADIUS_RATIO))
            rx2 = min(frame.shape[1], int(sx2 + sw * RESCUE_SEARCH_RADIUS_RATIO))
            ry2 = min(frame.shape[0], int(sy2 + sh * RESCUE_SEARCH_RADIUS_RATIO))
            search_crop = frame[ry1:ry2, rx1:rx2]
            if search_crop.size == 0:
                continue

            rescue_results = vehicle_model.predict(search_crop, conf=RESCUE_CONFIDENCE, imgsz=IMAGE_SIZE, verbose=False)
            for rbox in rescue_results[0].boxes:
                lx1, ly1, lx2, ly2 = rbox.xyxy[0].cpu().numpy()
                candidate_box = (lx1 + rx1, ly1 + ry1, lx2 + rx1, ly2 + ry1)  # map crop-local coords back to full frame

                # Skip if this is basically just the sudden-stop
                # vehicle re-detecting itself, or a vehicle the main
                # pass already found this frame (i.e. not actually a
                # miss - genuinely nothing new here).
                if compute_iou(candidate_box, sudden_stop_box) > 0.6:
                    continue
                already_tracked = any(
                    compute_iou(candidate_box, other_box) > 0.4
                    for other_id, other_box in box_by_track_id.items()
                    if other_id != sudden_stop_id
                )
                if already_tracked:
                    continue

                edge_dist = compute_edge_distance(candidate_box, sudden_stop_box)
                iou_with_sudden_stop = compute_iou(candidate_box, sudden_stop_box)
                if edge_dist > PROXIMITY_PIXEL_THRESHOLD and iou_with_sudden_stop < OVERLAP_IOU_THRESHOLD:
                    continue  # found some other vehicle nearby, but not actually touching this one

                r_class = vehicle_model.names.get(int(rbox.cls[0]), str(int(rbox.cls[0])))
                r_conf = float(rbox.conf[0])
                print(f"  [rescue] t={now:6.2f}s recovered a vehicle the main pass missed, next to "
                      f"track {sudden_stop_id} ({r_class}, conf={r_conf:.2f}) - checking as crash partner")

                cv2.rectangle(annotated_frame, (int(candidate_box[0]), int(candidate_box[1])),
                              (int(candidate_box[2]), int(candidate_box[3])), (0, 140, 255), 2)
                cv2.putText(annotated_frame, f"RESCUED {r_class} {r_conf:.2f}",
                            (int(candidate_box[0]), int(candidate_box[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

                rescue_union = union_box(candidate_box, sudden_stop_box)
                rescue_crop = padded_crop(frame, *rescue_union, CROP_PAD_RATIO)
                # Classify from motion first - the vehicle that
                # registered SUDDEN STOP has a real pre-crash speed,
                # and iou_with_sudden_stop (already computed above to
                # gate whether this candidate counts as touching it at
                # all) doubles as the peak-overlap signal. Vision is
                # still checked and can raise the tier further, but a
                # rescued partner no longer needs vision to fire before
                # anything gets written at all.
                # (No flew-away input here: this path writes once and
                # the rescued vehicle has no track to follow.)
                motion_tier = classify_motion_severity(
                    pre_stop_speed_kmh=last_sudden_stop_speed[sudden_stop_id],
                    peak_iou=iou_with_sudden_stop,
                )
                vision_tier, vision_conf = check_accident_visual(rescue_crop)
                new_tier = merge_severity(motion_tier, vision_tier)

                involved_tracks.add(sudden_stop_id)
                label_ids = [sudden_stop_id, f"rescued_{r_class}"]
                label_classes = [class_name_by_track_id[sudden_stop_id], r_class]
                accident_id = save_or_update_accident(
                    None, label_ids, label_classes, SEVERITY_DB_VALUES[new_tier],
                    detection_conf=r_conf, confirmation_conf=vision_conf,
                    now=now, crop_image=rescue_crop,
                )
                queue_event_gif(accident_id, "_".join(str(t) for t in label_ids), frame.shape,
                                    *rescue_union, now)
                if (now - last_alert_time[sudden_stop_id]) >= ALERT_COOLDOWN_SECONDS:
                    print(f"[ALERT] Track {sudden_stop_id} + rescued {r_class}: CONFIRMED "
                          f"{SEVERITY_LABELS[new_tier]} (recovered via rescue pass)")
                    last_alert_time[sudden_stop_id] = now
                break  # one rescued partner per sudden-stop vehicle is enough for this frame

        # ------------------------------------------------------
        # PASS 2: pairwise crash triggers. Two independent signals,
        # either one enough to start an event:
        #   (a) IoU overlap sustained for OVERLAP_FRAMES_REQUIRED frames
        #   (b) box edges within PROXIMITY_PIXEL_THRESHOLD px of each
        #       other AND at least one vehicle is in SUDDEN STOP,
        #       sustained for PROXIMITY_FRAMES_REQUIRED frames
        # All boxes for this frame are already drawn onto
        # annotated_frame above, so a crop of the union region here
        # shows both colored rectangles - visual proof of "these two
        # boxes are on top of each other."
        # ------------------------------------------------------
        current_ids = list(box_by_track_id.keys())
        iou_pairs_this_frame = set()
        proximity_pairs_this_frame = set()

        for i in range(len(current_ids)):
            for j in range(i + 1, len(current_ids)):
                id_a, id_b = current_ids[i], current_ids[j]
                if not (track_age_ok.get(id_a) and track_age_ok.get(id_b)):
                    continue  # don't trust overlap from a brand-new/just-switched track

                box_a = box_by_track_id[id_a]
                box_b = box_by_track_id[id_b]
                key = frozenset((int(id_a), int(id_b)))

                in_contact = False

                iou = compute_iou(box_a, box_b)
                if iou >= OVERLAP_IOU_THRESHOLD:
                    iou_pairs_this_frame.add(key)
                    overlap_counter[key] += 1
                    overlap_peak_iou[key] = max(overlap_peak_iou[key], iou)
                    in_contact = True

                edge_dist = compute_edge_distance(box_a, box_b)
                sudden_stop_nearby = (
                    status_by_track_id.get(id_a) == "SUDDEN STOP"
                    or status_by_track_id.get(id_b) == "SUDDEN STOP"
                )
                if edge_dist <= PROXIMITY_PIXEL_THRESHOLD and sudden_stop_nearby:
                    proximity_pairs_this_frame.add(key)
                    proximity_counter[key] += 1
                    in_contact = True

                # Snapshot each vehicle's position/speed/heading at first
                # contact so update_thrown_flag() (PASS 1, next frames)
                # can tell whether either one got knocked away.
                if in_contact:
                    note_contact(id_a, now)
                    note_contact(id_b, now)

        # Decay (don't hard-reset) the IoU counter for pairs that
        # aren't overlapping this exact frame, so one flickery frame
        # mid-crash doesn't throw away several seconds of accumulated
        # evidence. The proximity counter resets immediately instead -
        # it only needs a couple of frames to trust, so there's little
        # evidence to protect, and a hard reset keeps it reacting fast
        # to a genuine hard-brake-into-another-vehicle moment.
        for key in list(overlap_counter.keys()):
            if key not in iou_pairs_this_frame:
                overlap_counter[key] = max(0, overlap_counter[key] - OVERLAP_DECAY_PER_MISSED_FRAME)
                if overlap_counter[key] == 0:
                    del overlap_counter[key]

        for key in list(proximity_counter.keys()):
            if key not in proximity_pairs_this_frame:
                del proximity_counter[key]

        candidate_keys = set(overlap_counter.keys()) | set(proximity_counter.keys())

        for key in candidate_keys:
            count_iou = overlap_counter.get(key, 0)
            count_proximity = proximity_counter.get(key, 0)
            if count_iou < OVERLAP_FRAMES_REQUIRED and count_proximity < PROXIMITY_FRAMES_REQUIRED:
                continue
            id_a, id_b = tuple(key)
            if id_a not in box_by_track_id or id_b not in box_by_track_id:
                continue  # one of the pair wasn't detected this exact frame

            # Same duplicate-crash guard as the motion path: if this
            # pair hasn't already started its own event, and either
            # vehicle is already locked into a DIFFERENT confirmed
            # accident, don't spin up a brand-new event for it - it's
            # far more likely to be the already-crashed vehicle sitting
            # in the road overlapping the next car that drives past.
            # A pair that already has a row (already_started) is left
            # alone so its own event can keep being confirmed/updated.
            already_started = db_accident_id_for_pair[key] is not None
            if not already_started and (id_a in involved_tracks or id_b in involved_tracks):
                continue

            box_a = box_by_track_id[id_a]
            box_b = box_by_track_id[id_b]
            class_a = class_name_by_track_id[id_a]
            class_b = class_name_by_track_id[id_b]
            ux1, uy1, ux2, uy2 = union_box(box_a, box_b)

            if DEBUG_PRINT_STATUS_CHANGES and (count_iou == OVERLAP_FRAMES_REQUIRED or count_proximity == PROXIMITY_FRAMES_REQUIRED):
                trigger_reason = "IoU overlap" if count_iou >= OVERLAP_FRAMES_REQUIRED else "proximity + sudden stop"
                print(f"  [debug] t={now:6.2f}s OVERLAP trigger ({trigger_reason}): tracks {id_a}({class_a}) & "
                      f"{id_b}({class_b}) (iou_frames={count_iou}, proximity_frames={count_proximity})")

            if DEBUG_SAVE_TRIGGER_SNAPSHOTS and (now - last_overlap_debug_snapshot_time[key]) >= CONFIRMATION_CHECK_COOLDOWN:
                snap_path = os.path.join(
                    DEBUG_SNAPSHOT_DIR,
                    f"overlap_{id_a}_{id_b}_t{now:.2f}s.jpg"
                )
                cv2.imwrite(snap_path, annotated_frame)
                last_overlap_debug_snapshot_time[key] = now
                print(f"  [debug] overlap-only trigger snapshot saved: {snap_path}")

            # Log the event the moment either overlap trigger fires -
            # severity is classified straight from motion (peak IoU
            # reached so far, plus whichever vehicle's pre-crash speed
            # is highest) rather than left as a placeholder, so a real
            # tier shows up in the dashboard immediately.
            if db_accident_id_for_pair[key] is None:
                pending_crop = padded_crop(frame, ux1, uy1, ux2, uy2, CROP_PAD_RATIO)
                motion_tier = classify_motion_severity(
                    pre_stop_speed_kmh=max(
                        last_sudden_stop_speed[id_a], last_sudden_stop_speed[id_b]
                    ),
                    peak_iou=overlap_peak_iou[key],
                    flew_away=thrown_vehicle[id_a] or thrown_vehicle[id_b],
                )
                confirmed_overlap_severity[key] = motion_tier
                involved_tracks.update((id_a, id_b))  # this is a real, motion-classified event now, not a placeholder
                db_accident_id_for_pair[key] = save_or_update_accident(
                    None, [id_a, id_b], [class_a, class_b], SEVERITY_DB_VALUES[motion_tier],
                    # Overlap has no single YOLO detection confidence to
                    # report (IoU/proximity already gated this, not a
                    # class score).
                    detection_conf=0.99,
                    confirmation_conf=0.0,  # not yet corroborated by vision
                    now=now,
                    crop_image=pending_crop,
                )
                queue_event_gif(db_accident_id_for_pair[key], f"{id_a}_{id_b}", frame.shape,
                                    ux1, uy1, ux2, uy2, now)

            crop_for_model = None
            if (now - last_overlap_confirmation_check_time[key]) >= CONFIRMATION_CHECK_COOLDOWN:
                # Vision model checks the RAW frame region (no boxes
                # drawn on it) - cleaner input for the model.
                crop_for_model = padded_crop(frame, ux1, uy1, ux2, uy2, CROP_PAD_RATIO)
                vision_tier, vision_conf = check_accident_visual(crop_for_model)
                newly_vision_confirmed = vision_tier is not None and not vision_confirmed_overlap[key]
                if vision_tier is not None:
                    vision_confirmed_overlap[key] = True
                # An overlap-triggered pair may not have a clean
                # single-vehicle "sudden stop" speed (a T-bone can
                # happen without one) - re-classify from motion using
                # the peak IoU reached so far (it only grows over the
                # life of the event) plus whichever vehicle's pre-stop
                # speed (if any) is highest, then merge in vision.
                motion_tier = classify_motion_severity(
                    pre_stop_speed_kmh=max(
                        last_sudden_stop_speed[id_a], last_sudden_stop_speed[id_b]
                    ),
                    peak_iou=overlap_peak_iou[key],
                    flew_away=thrown_vehicle[id_a] or thrown_vehicle[id_b],
                )
                merged_tier = merge_severity(motion_tier, vision_tier)
                if DEBUG_PRINT_STATUS_CHANGES and merged_tier != confirmed_overlap_severity[key]:
                    print(f"  [debug] t={now:6.2f}s pair {id_a}&{id_b} severity re-check: "
                          f"stored={confirmed_overlap_severity[key]}, motion={motion_tier}, "
                          f"vision={vision_tier} (conf={vision_conf:.4f}), "
                          f"flew_away={thrown_vehicle[id_a] or thrown_vehicle[id_b]} -> "
                          f"{'kept (locked)' if LOCK_SEVERITY_AFTER_FIRST_CLASSIFICATION else 'updating to ' + merged_tier}")
                if LOCK_SEVERITY_AFTER_FIRST_CLASSIFICATION:
                    # Tier stays as first written. Just record vision's
                    # confidence once (no tier change, no extra crop).
                    if newly_vision_confirmed and db_accident_id_for_pair[key] is not None:
                        save_or_update_accident(
                            db_accident_id_for_pair[key], [id_a, id_b], [class_a, class_b],
                            SEVERITY_DB_VALUES[confirmed_overlap_severity[key]],
                            detection_conf=0.99,
                            confirmation_conf=vision_conf,
                            now=now,
                            crop_image=None,
                        )
                elif merged_tier != confirmed_overlap_severity[key]:
                    confirmed_overlap_severity[key] = merged_tier
                    # Vision agrees now - upgrade the SAME row instead
                    # of creating a second one for this event. The crop
                    # is the raw (no-boxes) frame region - a clean photo
                    # of the accident, not a debug overlay.
                    save_or_update_accident(
                        db_accident_id_for_pair[key], [id_a, id_b], [class_a, class_b],
                        SEVERITY_DB_VALUES[merged_tier],
                        detection_conf=0.99,
                        confirmation_conf=vision_conf,
                        now=now,
                        crop_image=crop_for_model,
                    )
                last_overlap_confirmation_check_time[key] = now

            severity = confirmed_overlap_severity[key]
            label = SEVERITY_LABELS[severity]
            if vision_confirmed_overlap[key]:
                label = f"CONFIRMED {label}"
                text_color = (255, 0, 255)   # magenta - motion AND vision agree
            else:
                label = f"{label} (motion)"
                text_color = TIER_DISPLAY_COLORS[severity]

            cv2.putText(
                annotated_frame, label,
                (int(ux1), max(15, int(uy1) - 40)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2
            )
            if (now - last_overlap_alert_time[key]) >= ALERT_COOLDOWN_SECONDS:
                print(f"[ALERT] Tracks {id_a} & {id_b} ({class_a}/{class_b}): "
                      f"{label} via box overlap")
                last_overlap_alert_time[key] = now

        # Once a pair stops qualifying under BOTH signals and its
        # counters fully clear (see above), the next time these same
        # two track_ids trigger again it's treated as a brand-new
        # event with its own database row.
        # (involved_tracks itself is NOT cleared here, on purpose -
        # once a track has been confirmed as crashed it stays locked
        # out of starting new events for the rest of the run.)
        for key in list(db_accident_id_for_pair.keys()):
            if key not in overlap_counter and key not in proximity_counter:
                db_accident_id_for_pair[key] = None
                confirmed_overlap_severity[key] = None
                vision_confirmed_overlap[key] = False
                overlap_peak_iou[key] = 0.0

    cv2.imshow("Vehicle Detection & Speed Estimation", annotated_frame)

    key_pressed = cv2.waitKey(delay) & 0xFF
    if key_pressed == 27:  # ESC
        break

    frame_idx += 1


cap.release()
cv2.destroyAllWindows()

# Any GIF still waiting on post-event footage gets built with whatever
# it has (video ended or ESC was pressed), then in-flight uploads finish.
flush_finished_gif_jobs(force=True)
pending_gifs = [t for t in gif_threads if t.is_alive()]
if pending_gifs:
    print(f"Waiting for {len(pending_gifs)} GIF upload(s) to finish...")
    for t in pending_gifs:
        t.join()

print("Detection stopped.")