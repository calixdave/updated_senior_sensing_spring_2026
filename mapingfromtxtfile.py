import os

# =========================================================
# MAP LOCATION FROM COMPACT LOCAL RESULT
# =========================================================
# Input:
#   results/compact_local_result.txt
#
# The compact local result is 16 characters:
#   8 surrounding cells x 2 chars each = 16
#
# Each pair:
#   floor color + object state
#
# Example:
#   PEYEBEMEPEBTMEBE
#
# Output:
#   results/compact_map_result.txt
#
# The output is 17 characters:
#   8 surrounding cells x 2 chars each = 16
#   final facing direction char = 1
# =========================================================

RESULTS_DIR = "results"

COMPACT_LOCAL_FILE = os.path.join(RESULTS_DIR, "compact_local_result.txt")
COMPACT_MAP_FILE = os.path.join(RESULTS_DIR, "compact_map_result.txt")

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

# Your scan order from runner:
# front -> right -> back -> left
SCAN_START_LOCAL = "FRONT"
SCAN_SWEEP = "cw"
NUM_VIEWS = 4


# =========================================================
# COMPACT LOCAL INPUT
# =========================================================

def read_compact_local_result(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")

    with open(path, "r") as f:
        compact = f.read().strip().replace(" ", "").replace("\n", "")

    if len(compact) != 16:
        raise ValueError(
            f"Expected 16 characters in compact local file, got {len(compact)}: {compact}"
        )

    compact = compact.upper()

    allowed_colors = {"P", "Y", "B", "M", "?"}
    allowed_objects = {"T", "O", "E", "?"}

    pairs = []

    for i in range(0, 16, 2):
        color_char = compact[i]
        object_char = compact[i + 1]

        if color_char not in allowed_colors:
            raise ValueError(f"Bad floor color character '{color_char}' at compact index {i}")

        if object_char not in allowed_objects:
            raise ValueError(f"Bad object character '{object_char}' at compact index {i + 1}")

        pairs.append((color_char, object_char))

    return pairs


def compact_pairs_to_local_grids(pairs):
    """
    Rebuild local 3x3 color and object matrices.

    Compact order from runner:
        top-left, top-middle, top-right,
        middle-left, middle-right,
        bottom-left, bottom-middle, bottom-right

    Center is inserted as A.
    """

    color = [
        ["?", "?", "?"],
        ["?", "A", "?"],
        ["?", "?", "?"],
    ]

    obj = [
        ["?", "?", "?"],
        ["?", "A", "?"],
        ["?", "?", "?"],
    ]

    positions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0),         (1, 2),
        (2, 0), (2, 1), (2, 2),
    ]

    for (r, c), (color_char, object_char) in zip(positions, pairs):
        color[r][c] = color_char
        obj[r][c] = object_char

    return color, obj


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

    window = [
        [grid[center_r - 1][center_c - 1], grid[center_r - 1][center_c], grid[center_r - 1][center_c + 1]],
        [grid[center_r][center_c - 1],     "A",                         grid[center_r][center_c + 1]],
        [grid[center_r + 1][center_c - 1], grid[center_r + 1][center_c], grid[center_r + 1][center_c + 1]],
    ]

    # Do not match windows touching unused/blocked cells.
    for r in range(3):
        for c in range(3):
            if window[r][c] == "X":
                return None

    return window


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
    Physical floor correction from your testing:
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
# COMPACT MAP OUTPUT
# =========================================================

def build_compact_17char(matched_biggrid_window, object_biggrid_perspective, final_direction_physical):
    """
    Output:
        8 surrounding cells x 2 chars each = 16
        final physical facing direction = 1

    Each non-center cell:
        floor color from matched BIG_GRID window + object state
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


def map_location_from_compact_local():
    pairs = read_compact_local_result(COMPACT_LOCAL_FILE)

    local_color_3x3, local_object_3x3 = compact_pairs_to_local_grids(pairs)

    best = find_best_match(local_color_3x3, BIG_GRID)

    if best is None:
        raise RuntimeError("No valid BIG_GRID match found from compact local result.")

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

    # Rotate object layer into BIG_GRID perspective using same rotation found
    # during color matching.
    object_biggrid_perspective = rotate_n_ccw(
        local_object_3x3,
        best["rot_steps"]
    )

    compact_map_result = build_compact_17char(
        best["matched_biggrid_window"],
        object_biggrid_perspective,
        camera_direction_after_scan_physical
    )

    return compact_map_result


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    compact_map_result = map_location_from_compact_local()

    with open(COMPACT_MAP_FILE, "w") as f:
        f.write(compact_map_result + "\n")

    # Print only final compact map result.
    print(compact_map_result)


if __name__ == "__main__":
    main()
