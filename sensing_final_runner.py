import os
import time
import cv2
import numpy as np

# =========================================================
# TEAM 4 SD26 FULL SENSING RUNNER
# =========================================================
# Pipeline:
#   1. Capture 4 still images: front, right, back, left
#   2. Detect floor colors: P/Y/B/M
#   3. Detect objects: green target, red obstacle
#   4. Match color 3x3 into BIG_GRID
#   5. Rotate object grid into BIG_GRID perspective
#   6. Save and print compact_map_result.txt only
# =========================================================

# =========================================================
# PATHS
# =========================================================

SCAN_DIR = "scan_images"
RESULTS_DIR = "results"
COMPACT_RESULT_FILE = os.path.join(RESULTS_DIR, "compact_map_result.txt")

# =========================================================
# CAMERA CONFIG
# =========================================================

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

HEADINGS = ["front", "right", "back", "left"]

# Number of frames to discard before saving, helps camera auto-exposure settle
WARMUP_FRAMES = 12

# Small delay after user presses Enter before capture
CAPTURE_DELAY_SEC = 0.3

# =========================================================
# ROI CONFIG
# =========================================================

ROI_TOP_FRAC = 0.55
ROI_BOT_FRAC = 0.95

SLOT_PAD_X_FRAC = 0.03
SLOT_PAD_Y_FRAC = 0.06

HEADING_TO_POSITIONS = {
    "front": [(-1, +1), (0, +1), (+1, +1)],
    "right": [(+1, +1), (+1, 0), (+1, -1)],
    "back":  [(+1, -1), (0, -1), (-1, -1)],
    "left":  [(-1, -1), (-1, 0), (-1, +1)],
}

# =========================================================
# OBJECT DETECTION CONFIG
# =========================================================
# Green box = Target
# Red box   = Obstacle
# Pink/magenta tile is rejected as fake red.
# =========================================================

GREEN_S_MIN = 70
GREEN_V_MIN = 70

RED_S_MIN = 110
RED_V_MIN = 80

RED_HUE1_LOW = 0
RED_HUE1_HIGH = 8
RED_HUE2_LOW = 176
RED_HUE2_HIGH = 180

PINK_HUE_LOW = 145
PINK_HUE_HIGH = 175
PINK_S_MIN = 60
PINK_V_MIN = 50
PINK_REJECT_RATIO = 0.20

MIN_OBJECT_RATIO = 0.025
MIN_CONTOUR_AREA_FRAC = 0.010
MAX_CONTOUR_AREA_FRAC = 0.65

MIN_OBJECT_W_FRAC = 0.08
MIN_OBJECT_H_FRAC = 0.08

MAX_CENTER_OFFSET = 0.95

# =========================================================
# BIG GRID
# =========================================================
# P = Purple
# Y = Yellow
# B = Blue
# M = Pink/Magenta
# X = blocked/unused
# =========================================================

BIG_GRID = [
    ["P", "Y", "Y", "Y", "M", "B", "M", "M", "P", "P"],
    ["P", "M", "Y", "P", "M", "B", "B", "M", "M", "X"],
    ["M", "Y", "P", "P", "B", "M", "M", "Y", "B", "X"],
    ["M", "Y", "M", "Y", "M", "B", "Y", "P", "Y", "Y"],
    ["M", "B", "M", "P", "P", "Y", "Y", "P", "B", "P"],
    ["B", "B", "P", "B", "B", "P", "B", "B", "B", "M"],
    ["Y", "P", "Y", "Y", "B", "B", "P", "P", "Y", "X"],
]

MIN_KNOWN_NEIGHBORS = 5
MAX_MISMATCHES = 3

SCAN_START_LOCAL = "FRONT"
SCAN_SWEEP = "cw"
NUM_VIEWS = 4


# =========================================================
# GENERAL HELPERS
# =========================================================

def ensure_clean_dirs():
    os.makedirs(SCAN_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Remove old scan images so every run uses only current images.
    for heading in HEADINGS:
        path = os.path.join(SCAN_DIR, f"{heading}.jpg")
        if os.path.exists(path):
            os.remove(path)

    # Remove old compact result.
    if os.path.exists(COMPACT_RESULT_FILE):
        os.remove(COMPACT_RESULT_FILE)


def make_empty_local_grid():
    return {
        (-1, +1): "?",
        (0, +1): "?",
        (+1, +1): "?",
        (-1, 0): "?",
        (0, 0): "A",
        (+1, 0): "?",
        (-1, -1): "?",
        (0, -1): "?",
        (+1, -1): "?",
    }


def local_grid_to_matrix(final_grid):
    rows = []

    for row in [1, 0, -1]:
        vals = []
        for col in [-1, 0, 1]:
            vals.append(final_grid.get((col, row), "?"))
        rows.append(vals)

    return rows


def pretty_matrix(mat):
    return "\n".join(" ".join(row) for row in mat)


# =========================================================
# STEP 1: SCAN CAPTURE
# =========================================================

def capture_scan_images():
    """
    Captures 4 still images with no camera preview.
    Robot/Camera manually rotated between captures durint test.
    Press Enter in the terminal for each heading.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    try:
        for heading in HEADINGS:
            input(f"Point camera to {heading.upper()}, then press Enter to capture... ")

            time.sleep(CAPTURE_DELAY_SEC)

            frame = None

            for _ in range(WARMUP_FRAMES):
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError(f"Failed to read frame for heading: {heading}")

            filename = os.path.join(SCAN_DIR, f"{heading}.jpg")
            ok = cv2.imwrite(filename, frame)

            if not ok:
                raise RuntimeError(f"Failed to save image: {filename}")

    finally:
        cap.release()


# =========================================================
# ROI EXTRACTION
# =========================================================

def get_three_slot_rois(img):
    h, w = img.shape[:2]

    y0 = int(ROI_TOP_FRAC * h)
    y1 = int(ROI_BOT_FRAC * h)

    if y1 <= y0:
        return []

    band = img[y0:y1, :]
    bh, bw = band.shape[:2]

    slots = []

    for i in range(3):
        sx0 = int(i * bw / 3)
        sx1 = int((i + 1) * bw / 3)

        pad_x = int(SLOT_PAD_X_FRAC * (sx1 - sx0))
        pad_y = int(SLOT_PAD_Y_FRAC * bh)

        cx0 = max(0, sx0 + pad_x)
        cx1 = min(bw, sx1 - pad_x)
        cy0 = max(0, pad_y)
        cy1 = min(bh, bh - pad_y)

        crop = band[cy0:cy1, cx0:cx1]
        slots.append(crop)

    return slots


def center_crop(img, frac=0.55):
    h, w = img.shape[:2]

    y0 = int((1.0 - frac) * 0.5 * h)
    y1 = int(h - y0)

    x0 = int((1.0 - frac) * 0.5 * w)
    x1 = int(w - x0)

    return img[y0:y1, x0:x1]


# =========================================================
# STEP 2: FLOOR COLOR DETECTION
# =========================================================

def classify_floor_color_opencv(tile_bgr):
    """
    Detects only fixed floor colors:
        P = purple
        Y = yellow
        B = blue
        M = pink/magenta
    Red and green are reserved for objects.
    """
    if tile_bgr is None or tile_bgr.size == 0:
        return "?"

    roi = center_crop(tile_bgr, 0.55)

    if roi.size == 0:
        return "?"

    roi = cv2.GaussianBlur(roi, (5, 5), 0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    A = lab[:, :, 1]
    LB = lab[:, :, 2]

    valid = (S >= 45) & (V >= 45)

    valid_ratio = float(np.count_nonzero(valid)) / float(valid.size)

    if valid_ratio < 0.10:
        return "?"

    yellow_mask = ((H >= 18) & (H <= 38) & valid)
    blue_mask = ((H >= 95) & (H <= 135) & valid)
    purple_pink_mask = ((H >= 136) & (H <= 169) & valid)

    ratios = {
        "yellow": float(np.count_nonzero(yellow_mask)) / float(valid.size),
        "blue": float(np.count_nonzero(blue_mask)) / float(valid.size),
        "purple_pink": float(np.count_nonzero(purple_pink_mask)) / float(valid.size),
    }

    h_mean = float(np.mean(H[valid]))
    v_mean = float(np.mean(V[valid]))
    a_mean = float(np.mean(A[valid]))
    lb_mean = float(np.mean(LB[valid]))

    best_basic = max(ratios, key=lambda k: ratios[k])
    best_ratio = ratios[best_basic]

    if best_ratio >= 0.15:
        if best_basic == "yellow":
            return "Y"

        if best_basic == "blue":
            return "B"

        if best_basic == "purple_pink":
            # Pink/magenta tends to be brighter/warmer than purple.
            if v_mean >= 125 or lb_mean >= 145:
                return "M"
            else:
                return "P"

    # Fallback for pink/purple if hue mask is weak.
    if a_mean >= 145 and lb_mean >= 135 and v_mean >= 105:
        return "M"

    if a_mean >= 140 and lb_mean < 140:
        return "P"

    # Extra fallback by hue average.
    if 18 <= h_mean <= 38:
        return "Y"

    if 95 <= h_mean <= 135:
        return "B"

    return "?"


def detect_floor_colors_from_images():
    local_grid = make_empty_local_grid()

    for heading in HEADINGS:
        path = os.path.join(SCAN_DIR, f"{heading}.jpg")

        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")

        slots = get_three_slot_rois(img)
        if len(slots) != 3:
            raise RuntimeError(f"Could not build 3 slots for heading: {heading}")

        for i, tile in enumerate(slots):
            ch = classify_floor_color_opencv(tile)
            pos = HEADING_TO_POSITIONS[heading][i]
            local_grid[pos] = ch

    return local_grid_to_matrix(local_grid)


# =========================================================
# STEP 3: OBJECT DETECTION
# =========================================================

def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def build_red_green_pink_masks(slot_bgr):
    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    red_mask_1 = (
        (H >= RED_HUE1_LOW) &
        (H <= RED_HUE1_HIGH) &
        (S >= RED_S_MIN) &
        (V >= RED_V_MIN)
    )

    red_mask_2 = (
        (H >= RED_HUE2_LOW) &
        (H <= RED_HUE2_HIGH) &
        (S >= RED_S_MIN) &
        (V >= RED_V_MIN)
    )

    red_mask = (red_mask_1 | red_mask_2).astype(np.uint8) * 255

    green_mask = (
        (H >= 40) &
        (H <= 90) &
        (S >= GREEN_S_MIN) &
        (V >= GREEN_V_MIN)
    ).astype(np.uint8) * 255

    pink_reject_mask = (
        (H >= PINK_HUE_LOW) &
        (H <= PINK_HUE_HIGH) &
        (S >= PINK_S_MIN) &
        (V >= PINK_V_MIN)
    ).astype(np.uint8) * 255

    red_mask = clean_mask(red_mask)
    green_mask = clean_mask(green_mask)
    pink_reject_mask = clean_mask(pink_reject_mask)

    return red_mask, green_mask, pink_reject_mask


def get_largest_valid_blob(mask, slot_shape):
    h, w = slot_shape[:2]
    slot_area = h * w

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area <= 0:
            continue

        area_frac = area / float(slot_area)

        if area_frac < MIN_CONTOUR_AREA_FRAC:
            continue

        if area_frac > MAX_CONTOUR_AREA_FRAC:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        w_frac = bw / float(w)
        h_frac = bh / float(h)

        if w_frac < MIN_OBJECT_W_FRAC:
            continue

        if h_frac < MIN_OBJECT_H_FRAC:
            continue

        cx = x + bw / 2.0
        cy = y + bh / 2.0

        center_dx = abs(cx - w / 2.0) / max(1.0, w / 2.0)
        center_dy = abs(cy - h / 2.0) / max(1.0, h / 2.0)

        if center_dx > MAX_CENTER_OFFSET:
            continue

        if center_dy > MAX_CENTER_OFFSET:
            continue

        if area > best_area:
            best_area = area
            best = {
                "area": area,
                "area_frac": area_frac,
                "bbox": (x, y, bw, bh),
            }

    return best


def detect_one_object_slot(slot_bgr):
    """
    Return:
        T = green target
        O = red obstacle
        E = empty
        ? = uncertain
    """
    if slot_bgr is None or slot_bgr.size == 0:
        return "?"

    red_mask, green_mask, pink_reject_mask = build_red_green_pink_masks(slot_bgr)

    red_ratio = float(np.count_nonzero(red_mask)) / float(red_mask.size)
    green_ratio = float(np.count_nonzero(green_mask)) / float(green_mask.size)
    pink_ratio = float(np.count_nonzero(pink_reject_mask)) / float(pink_reject_mask.size)

    red_blob = get_largest_valid_blob(red_mask, slot_bgr.shape)
    green_blob = get_largest_valid_blob(green_mask, slot_bgr.shape)

    red_detected = red_ratio >= MIN_OBJECT_RATIO and red_blob is not None
    green_detected = green_ratio >= MIN_OBJECT_RATIO and green_blob is not None

    # Reject fake red from pink/magenta floor tile.
    if pink_ratio >= PINK_REJECT_RATIO and red_ratio < 0.08:
        red_detected = False

    if pink_ratio > (red_ratio * 2.5) and red_ratio < 0.12:
        red_detected = False

    if red_detected and green_detected:
        if red_ratio > green_ratio:
            return "O"
        else:
            return "T"

    if red_detected:
        return "O"

    if green_detected:
        return "T"

    return "E"


def detect_objects_from_images():
    local_grid = make_empty_local_grid()

    for heading in HEADINGS:
        path = os.path.join(SCAN_DIR, f"{heading}.jpg")

        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")

        slots = get_three_slot_rois(img)
        if len(slots) != 3:
            raise RuntimeError(f"Could not build 3 slots for heading: {heading}")

        for i, tile in enumerate(slots):
            obj = detect_one_object_slot(tile)
            pos = HEADING_TO_POSITIONS[heading][i]
            local_grid[pos] = obj

    return local_grid_to_matrix(local_grid)


# =========================================================
# STEP 4: MAP LOCATION
# =========================================================

def rotate_3x3_ccw(mat):
    return [
        [mat[0][2], mat[1][2], mat[2][2]],
        [mat[0][1], mat[1][1], mat[2][1]],
        [mat[0][0], mat[1][0], mat[2][0]],
    ]


def rotate_n_ccw(mat, n):
    out = [row[:] for row in mat]

    for _ in range(n % 4):
        out = rotate_3x3_ccw(out)

    return out


def get_window_3x3(grid, center_r, center_c):
    rows = len(grid)
    cols = len(grid[0])

    if center_r - 1 < 0 or center_r + 1 >= rows:
        return None

    if center_c - 1 < 0 or center_c + 1 >= cols:
        return None

    raw = [
        [grid[center_r - 1][center_c - 1], grid[center_r - 1][center_c], grid[center_r - 1][center_c + 1]],
        [grid[center_r][center_c - 1],     "A",                         grid[center_r][center_c + 1]],
        [grid[center_r + 1][center_c - 1], grid[center_r + 1][center_c], grid[center_r + 1][center_c + 1]],
    ]

    for r in range(3):
        for c in range(3):
            if raw[r][c] == "X":
                return None

    return raw


def score_match(local_3x3, window_3x3):
    known = 0
    matches = 0
    mismatches = 0

    for r in range(3):
        for c in range(3):
            lv = local_3x3[r][c]
            wv = window_3x3[r][c]

            if lv == "A":
                continue

            if lv == "?":
                continue

            known += 1

            if lv == wv:
                matches += 1
            else:
                mismatches += 1

    return {
        "known": known,
        "matches": matches,
        "mismatches": mismatches,
        "score": matches,
    }


def rotation_to_facing(rotation_ccw_deg):
    """
    Raw matrix convention.
    """
    mapping = {
        0: "UP",
        90: "RIGHT",
        180: "DOWN",
        270: "LEFT",
    }

    return mapping[rotation_ccw_deg]


def physical_direction_fix(direction):
    """
    Physical floor correction:
        raw UP   -> physical DOWN
        raw DOWN -> physical UP
        RIGHT/LEFT unchanged
    """
    mapping = {
        "UP": "DOWN",
        "DOWN": "UP",
        "RIGHT": "RIGHT",
        "LEFT": "LEFT",
    }

    return mapping[direction]


def rotate_direction(direction, steps_ccw):
    dirs = ["UP", "LEFT", "DOWN", "RIGHT"]
    idx = dirs.index(direction)

    return dirs[(idx + steps_ccw) % 4]


def get_scan_order(scan_start_local="FRONT", scan_sweep="cw", num_views=4):
    scan_start_local = scan_start_local.upper()
    scan_sweep = scan_sweep.lower()

    if scan_sweep == "cw":
        base_order = ["FRONT", "RIGHT", "BACK", "LEFT"]
    elif scan_sweep == "ccw":
        base_order = ["FRONT", "LEFT", "BACK", "RIGHT"]
    else:
        raise ValueError(f"scan_sweep must be 'cw' or 'ccw', got: {scan_sweep}")

    if scan_start_local not in base_order:
        raise ValueError(f"scan_start_local must be one of {base_order}, got: {scan_start_local}")

    start_idx = base_order.index(scan_start_local)
    ordered = base_order[start_idx:] + base_order[:start_idx]

    return ordered[:num_views]


def local_heading_to_map_direction(start_map_direction, local_heading):
    local_heading = local_heading.upper()

    local_steps_ccw = {
        "FRONT": 0,
        "LEFT": 1,
        "BACK": 2,
        "RIGHT": 3,
    }

    return rotate_direction(start_map_direction, local_steps_ccw[local_heading])


def get_final_camera_direction_after_scan(
    start_map_direction,
    scan_start_local="FRONT",
    scan_sweep="cw",
    num_views=4
):
    order = get_scan_order(scan_start_local, scan_sweep, num_views)
    final_local_heading = order[-1]

    final_map_direction = local_heading_to_map_direction(
        start_map_direction,
        final_local_heading
    )

    return final_local_heading, final_map_direction


def direction_to_char(direction):
    mapping = {
        "UP": "U",
        "RIGHT": "R",
        "DOWN": "D",
        "LEFT": "L",
    }

    return mapping[direction]


def find_best_match(local_color_3x3, big_grid):
    rows = len(big_grid)
    cols = len(big_grid[0])

    candidates = []

    for rot_steps in range(4):
        rotated_local_color = rotate_n_ccw(local_color_3x3, rot_steps)
        rotation_ccw_deg = rot_steps * 90

        for center_r in range(1, rows - 1):
            for center_c in range(1, cols - 1):
                window = get_window_3x3(big_grid, center_r, center_c)

                if window is None:
                    continue

                s = score_match(rotated_local_color, window)

                if s["known"] < MIN_KNOWN_NEIGHBORS:
                    continue

                if s["mismatches"] > MAX_MISMATCHES:
                    continue

                candidates.append({
                    "center_row": center_r,
                    "center_col": center_c,
                    "rot_steps": rot_steps,
                    "rotation_ccw_deg": rotation_ccw_deg,
                    "facing_raw": rotation_to_facing(rotation_ccw_deg),
                    "known": s["known"],
                    "matches": s["matches"],
                    "mismatches": s["mismatches"],
                    "score": s["score"],
                    "rotated_local_color": rotated_local_color,
                    "matched_biggrid_window": window,
                })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (x["score"], -x["mismatches"], x["known"]),
        reverse=True
    )

    return candidates[0]


def build_compact_17char(matched_biggrid_window, object_biggrid_perspective, final_direction_physical):
    """
    17-character compact output:
        8 surrounding cells x 2 chars each = 16
        final physical direction char = 1

    Each non-center cell:
        floor color + object state

    Example:
        YE = Yellow tile, Empty object
        PO = Purple tile, Obstacle
        BT = Blue tile, Target
    """
    out = []

    for r in range(3):
        for c in range(3):
            if r == 1 and c == 1:
                continue

            floor_char = str(matched_biggrid_window[r][c]).strip().upper()[:1]
            obj_char = str(object_biggrid_perspective[r][c]).strip().upper()[:1]

            if obj_char not in ["T", "O", "E", "?"]:
                obj_char = "?"

            out.append(floor_char + obj_char)

    out.append(direction_to_char(final_direction_physical))

    return "".join(out)


def map_location_and_build_compact(local_color_3x3, local_object_3x3):
    best = find_best_match(local_color_3x3, BIG_GRID)

    if best is None:
        raise RuntimeError("No valid BIG_GRID match found.")

    camera_direction_before_scan_raw = best["facing_raw"]

    _, camera_direction_after_scan_raw = get_final_camera_direction_after_scan(
        start_map_direction=camera_direction_before_scan_raw,
        scan_start_local=SCAN_START_LOCAL,
        scan_sweep=SCAN_SWEEP,
        num_views=NUM_VIEWS
    )

    camera_direction_after_scan_physical = physical_direction_fix(
        camera_direction_after_scan_raw
    )

    # Rotate object layer into BIG_GRID perspective using same color-match rotation.
    object_biggrid_perspective = rotate_n_ccw(
        local_object_3x3,
        best["rot_steps"]
    )

    compact = build_compact_17char(
        best["matched_biggrid_window"],
        object_biggrid_perspective,
        camera_direction_after_scan_physical
    )

    return compact


# =========================================================
# MAIN
# =========================================================

def main():
    ensure_clean_dirs()

    capture_scan_images()

    local_color_3x3 = detect_floor_colors_from_images()
    local_object_3x3 = detect_objects_from_images()

    compact_result = map_location_and_build_compact(
        local_color_3x3,
        local_object_3x3
    )

    with open(COMPACT_RESULT_FILE, "w") as f:
        f.write(compact_result + "\n" + "\n")

    # Print only the compact result.
    print(compact_result)


if __name__ == "__main__":
    main()
