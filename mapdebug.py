import os

# =========================================================
# CONFIG
# =========================================================

COLOR_FILE = "results/local_color_3x3.txt"
OBJECT_FILE = "results/local_object_3x3.txt"

RESULTS_DIR = "results"

MAP_RESULT_FILE = os.path.join(RESULTS_DIR, "map_result.txt")
COMPACT_RESULT_FILE = os.path.join(RESULTS_DIR, "compact_map_result.txt")
FINAL_GRID_FILE = os.path.join(RESULTS_DIR, "final_mapped_3x3.txt")
FINAL_OBJECT_GRID_FILE = os.path.join(RESULTS_DIR, "final_object_3x3_biggrid_perspective.txt")

# =========================================================
# BIG GRID
# =========================================================
# P = Purple
# Y = Yellow
# B = Blue
# M = Pink/Magenta
# X = blocked/unused
#
# 7 x 10 = 70 cells
# 67 colored tiles + 3 blocked cells
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

# Matching tolerance
MIN_KNOWN_NEIGHBORS = 5
MAX_MISMATCHES = 3

# Your scan order:
# front -> right -> back -> left
SCAN_START_LOCAL = "FRONT"
SCAN_SWEEP = "cw"
NUM_VIEWS = 4


# =========================================================
# FILE / MATRIX HELPERS
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


def pretty_matrix(mat):
    return "\n".join(" ".join(row) for row in mat)


def save_text(path, text):
    with open(path, "w") as f:
        f.write(text)


# =========================================================
# ROTATION HELPERS
# =========================================================

def rotate_3x3_ccw(mat):
    """
    Rotate a 3x3 matrix counterclockwise.

    Example:
        a b c
        d e f
        g h i

    becomes:
        c f i
        b e h
        a d g
    """
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
# BIG GRID WINDOW HELPERS
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

    # Ignore any 3x3 candidate that includes a blocked/unused tile.
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


# =========================================================
# DIRECTION HELPERS
# =========================================================

def rotation_to_facing(rotation_ccw_deg):
    """
    Raw matrix convention.

    This is kept as the original matching convention.
    Do not change this if the 3x3 matching works correctly.
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
    Physical floor convention fix.

    You observed:
        raw UP    should be physical DOWN
        raw DOWN  should be physical UP
        RIGHT and LEFT are already correct

    So only UP/DOWN are swapped.
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
# OBJECT PLACEMENT HELPERS
# =========================================================

def convert_object_grid_to_biggrid_perspective(local_object_3x3, rot_steps):
    """
    The object detector produces objects in the robot/local 3x3 view.

    After map matching, the color grid may be rotated to match BIG_GRID.
    Therefore, the object grid must be rotated by the SAME rot_steps.

    This gives object positions in the BIG_GRID / best-match-window perspective.
    """
    return rotate_n_ccw(local_object_3x3, rot_steps)


def overlay_objects_on_matched_window(matched_biggrid_window, object_biggrid_perspective):
    """
    Create final displayed 3x3 in BIG_GRID perspective.

    matched_biggrid_window:
        actual 3x3 floor window from BIG_GRID

    object_biggrid_perspective:
        object layer after rotating it into BIG_GRID perspective

    Object layer rules:
        T = target overrides floor
        O = obstacle overrides floor
        E = empty; keep floor color
        ? = unknown; keep floor color
        A = agent center
    """
    out = []

    for r in range(3):
        row = []

        for c in range(3):
            floor_char = matched_biggrid_window[r][c]
            obj_char = object_biggrid_perspective[r][c]

            if floor_char == "A":
                row.append("A")
            elif obj_char == "T":
                row.append("T")
            elif obj_char == "O":
                row.append("O")
            else:
                row.append(floor_char)

        out.append(row)

    return out


def build_compact_17char(matched_biggrid_window, object_biggrid_perspective, final_direction_physical):
    """
    17-character compact output:
        8 surrounding cells x 2 chars each = 16
        final physical direction char = 1

    Each surrounding cell gives:
        floor color + object state

    Example:
        YE means Yellow floor, Empty object
        PO means Purple floor, Obstacle
        BT means Blue floor, Target

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


# =========================================================
# MATCHING
# =========================================================

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
        return None, []

    candidates.sort(
        key=lambda x: (x["score"], -x["mismatches"], x["known"]),
        reverse=True
    )

    return candidates[0], candidates


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    local_color_3x3 = read_local_3x3(COLOR_FILE)
    local_object_3x3 = read_local_3x3(OBJECT_FILE)

    print("\nLOCAL COLOR 3x3 FROM COLOR DETECTION:")
    print(pretty_matrix(local_color_3x3))

    print("\nLOCAL OBJECT 3x3 FROM OBJECT DETECTION:")
    print(pretty_matrix(local_object_3x3))

    best, candidates = find_best_match(local_color_3x3, BIG_GRID)

    if best is None:
        msg = (
            "No valid match found.\n"
            "Try:\n"
            "- lowering MIN_KNOWN_NEIGHBORS\n"
            "- increasing MAX_MISMATCHES\n"
            "- checking that local_color_3x3.txt uses only P/Y/B/M/A/?\n"
            "- improving color detection\n"
        )

        print("\n" + msg)

        save_text(MAP_RESULT_FILE, msg + "\n")
        print(f"Saved: {MAP_RESULT_FILE}")
        return

    # Raw direction from matrix/matching convention
    camera_direction_before_scan_raw = best["facing_raw"]

    final_local_heading_after_scan, camera_direction_after_scan_raw = get_final_camera_direction_after_scan(
        start_map_direction=camera_direction_before_scan_raw,
        scan_start_local=SCAN_START_LOCAL,
        scan_sweep=SCAN_SWEEP,
        num_views=NUM_VIEWS
    )

    # Physical corrected direction for your floor observation
    camera_direction_before_scan_physical = physical_direction_fix(camera_direction_before_scan_raw)
    camera_direction_after_scan_physical = physical_direction_fix(camera_direction_after_scan_raw)

    # =====================================================
    # IMPORTANT OBJECT UPDATE
    # =====================================================
    # Convert local object result into BIG_GRID perspective
    # using the exact same rotation that matched the color grid.
    # =====================================================

    object_biggrid_perspective = convert_object_grid_to_biggrid_perspective(
        local_object_3x3,
        best["rot_steps"]
    )

    final_mapped_3x3 = overlay_objects_on_matched_window(
        best["matched_biggrid_window"],
        object_biggrid_perspective
    )

    compact_17char = build_compact_17char(
        best["matched_biggrid_window"],
        object_biggrid_perspective,
        camera_direction_after_scan_physical
    )

    print("\nBEST MATCH FOUND")
    print("-------------------------")
    print(f"center_row                          = {best['center_row']}")
    print(f"center_col                          = {best['center_col']}")
    print(f"rotation_ccw_deg                    = {best['rotation_ccw_deg']}")
    print(f"facing_raw_matrix                   = {camera_direction_before_scan_raw}")
    print(f"facing_physical_biggrid             = {camera_direction_before_scan_physical}")
    print(f"scan_start_local                    = {SCAN_START_LOCAL}")
    print(f"scan_sweep                          = {SCAN_SWEEP}")
    print(f"final_local_heading_after_scan      = {final_local_heading_after_scan}")
    print(f"camera_direction_after_scan_raw     = {camera_direction_after_scan_raw}")
    print(f"camera_direction_after_scan_physical= {camera_direction_after_scan_physical}")
    print(f"compact_17char                      = {compact_17char}")
    print(f"known_neighbors                     = {best['known']}")
    print(f"matches                             = {best['matches']}")
    print(f"mismatches                          = {best['mismatches']}")
    print(f"score                               = {best['score']}/{best['known']}")

    print("\nROTATED LOCAL COLOR 3x3 USED FOR MATCH:")
    print(pretty_matrix(best["rotated_local_color"]))

    print("\nMATCHED BIG_GRID WINDOW:")
    print(pretty_matrix(best["matched_biggrid_window"]))

    print("\nLOCAL OBJECT 3x3 BEFORE ROTATION:")
    print(pretty_matrix(local_object_3x3))

    print("\nOBJECT 3x3 AFTER ROTATION INTO BIG_GRID PERSPECTIVE:")
    print(pretty_matrix(object_biggrid_perspective))

    print("\nFINAL 3x3 IN BIG_GRID PERSPECTIVE WITH OBJECTS PLACED:")
    print(pretty_matrix(final_mapped_3x3))

    print(f"\nTotal valid candidates: {len(candidates)}")

    if len(candidates) > 1:
        print("\nTop candidates:")

        for i, c in enumerate(candidates[:5], start=1):
            raw_dir = c["facing_raw"]
            physical_dir = physical_direction_fix(raw_dir)

            print(
                f"{i}. center=({c['center_row']},{c['center_col']}), "
                f"rot={c['rotation_ccw_deg']} deg, "
                f"raw_facing={raw_dir}, "
                f"physical_facing={physical_dir}, "
                f"score={c['score']}/{c['known']}, "
                f"mismatches={c['mismatches']}"
            )

    result_lines = [
        "BEST MATCH FOUND",
        "-------------------------",
        f"center_row                          = {best['center_row']}",
        f"center_col                          = {best['center_col']}",
        f"rotation_ccw_deg                    = {best['rotation_ccw_deg']}",
        f"facing_raw_matrix                   = {camera_direction_before_scan_raw}",
        f"facing_physical_biggrid             = {camera_direction_before_scan_physical}",
        f"scan_start_local                    = {SCAN_START_LOCAL}",
        f"scan_sweep                          = {SCAN_SWEEP}",
        f"final_local_heading_after_scan      = {final_local_heading_after_scan}",
        f"camera_direction_after_scan_raw     = {camera_direction_after_scan_raw}",
        f"camera_direction_after_scan_physical= {camera_direction_after_scan_physical}",
        f"compact_17char                      = {compact_17char}",
        f"known_neighbors                     = {best['known']}",
        f"matches                             = {best['matches']}",
        f"mismatches                          = {best['mismatches']}",
        f"score                               = {best['score']}/{best['known']}",
        "",
        "LOCAL COLOR 3x3 FROM COLOR DETECTION:",
        pretty_matrix(local_color_3x3),
        "",
        "ROTATED LOCAL COLOR 3x3 USED FOR MATCH:",
        pretty_matrix(best["rotated_local_color"]),
        "",
        "MATCHED BIG_GRID WINDOW:",
        pretty_matrix(best["matched_biggrid_window"]),
        "",
        "LOCAL OBJECT 3x3 BEFORE ROTATION:",
        pretty_matrix(local_object_3x3),
        "",
        "OBJECT 3x3 AFTER ROTATION INTO BIG_GRID PERSPECTIVE:",
        pretty_matrix(object_biggrid_perspective),
        "",
        "FINAL 3x3 IN BIG_GRID PERSPECTIVE WITH OBJECTS PLACED:",
        pretty_matrix(final_mapped_3x3),
        "",
        f"Total valid candidates: {len(candidates)}",
    ]

    if len(candidates) > 1:
        result_lines.append("")
        result_lines.append("Top candidates:")

        for i, c in enumerate(candidates[:5], start=1):
            raw_dir = c["facing_raw"]
            physical_dir = physical_direction_fix(raw_dir)

            result_lines.append(
                f"{i}. center=({c['center_row']},{c['center_col']}), "
                f"rot={c['rotation_ccw_deg']} deg, "
                f"raw_facing={raw_dir}, "
                f"physical_facing={physical_dir}, "
                f"score={c['score']}/{c['known']}, "
                f"mismatches={c['mismatches']}"
            )

    with open(MAP_RESULT_FILE, "w") as f:
        f.write("\n".join(result_lines) + "\n")

    with open(COMPACT_RESULT_FILE, "w") as f:
        f.write(compact_17char + "\n")

    with open(FINAL_GRID_FILE, "w") as f:
        f.write(pretty_matrix(final_mapped_3x3) + "\n")

    with open(FINAL_OBJECT_GRID_FILE, "w") as f:
        f.write(pretty_matrix(object_biggrid_perspective) + "\n")

    print(f"\nSaved: {MAP_RESULT_FILE}")
    print(f"Saved: {COMPACT_RESULT_FILE}")
    print(f"Saved: {FINAL_GRID_FILE}")
    print(f"Saved: {FINAL_OBJECT_GRID_FILE}")


if __name__ == "__main__":
    main()
