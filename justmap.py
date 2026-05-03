import os

# =========================================================
# PI-FRIENDLY MAP LOCATION ONLY
# =========================================================
# Input:
#   results/local_color_3x3.txt
#   results/local_object_3x3.txt
#
# Output only:
#   results/compact_map_result.txt
#
# Uses:
#   - Correct 7x10 BIG_GRID
#   - Object rotation into BIG_GRID perspective
#   - UP/DOWN physical direction fix
# =========================================================

COLOR_FILE = "results/local_color_3x3.txt"
OBJECT_FILE = "results/local_object_3x3.txt"

RESULTS_DIR = "results"
COMPACT_RESULT_FILE = os.path.join(RESULTS_DIR, "compact_map_result.txt")

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

# Scan order used by your capture:
# front -> right -> back -> left
SCAN_START_LOCAL = "FRONT"
SCAN_SWEEP = "cw"
NUM_VIEWS = 4


# =========================================================
# FILE HELPERS
# =========================================================

def read_local_3x3(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")

    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.replace(",", " ").split()

            if len(parts) != 3:
                raise ValueError(f"Each row must have 3 entries. Bad row: {line}")

            rows.append([p.upper() for p in parts])

    if len(rows) != 3:
        raise ValueError(f"Expected 3 rows in local 3x3 file, found {len(rows)}")

    return rows


# =========================================================
# ROTATION HELPERS
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


# =========================================================
# BIG GRID MATCHING
# =========================================================

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

    # Do not match windows touching blocked/unused cells.
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
                    "matched_biggrid_window": window,
                })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (x["score"], -x["mismatches"], x["known"]),
        reverse=True
    )

    return candidates[0]


# =========================================================
# DIRECTION HELPERS
# =========================================================

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


# =========================================================
# COMPACT RESULT
# =========================================================

def build_compact_17char(matched_biggrid_window, object_biggrid_perspective, final_direction_physical):
    """
    17-character compact output:
        8 surrounding cells x 2 chars each = 16
        final physical direction char = 1

    Each non-center cell:
        floor color + object state

    Examples:
        YE = Yellow floor, Empty object
        PO = Purple floor, Obstacle
        BT = Blue floor, Target

    Center A is skipped.
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

    # Rotate object layer into BIG_GRID perspective using the same rotation
    # found during color matching.
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
    os.makedirs(RESULTS_DIR, exist_ok=True)

    local_color_3x3 = read_local_3x3(COLOR_FILE)
    local_object_3x3 = read_local_3x3(OBJECT_FILE)

    compact_result = map_location_and_build_compact(
        local_color_3x3,
        local_object_3x3
    )

    with open(COMPACT_RESULT_FILE, "w") as f:
        f.write(compact_result + "\n")

    # Print only compact result.
    print(compact_result)


if __name__ == "__main__":
    main()
