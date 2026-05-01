import os
import json
import cv2
import numpy as np

# =========================================================
# CONFIG
# =========================================================

SCAN_DIR = "scan_images"
DEBUG_DIR = "debug_tiles"
RESULTS_DIR = "results"

HEADINGS = ["front", "right", "back", "left"]

# Lower ROI to focus on first row of tiles
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


def matrix_rows_from_grid(final_grid):
    rows = []

    for row in [1, 0, -1]:
        vals = []
        for col in [-1, 0, 1]:
            vals.append(final_grid.get((col, row), "?"))
        rows.append(vals)

    return rows


def pretty_print_matrix(final_grid):
    rows = matrix_rows_from_grid(final_grid)

    for row in rows:
        print(" ".join(row))


def save_matrix_txt(path, final_grid):
    rows = matrix_rows_from_grid(final_grid)

    with open(path, "w") as f:
        for row in rows:
            f.write(" ".join(row) + "\n")


def center_crop(img, frac=0.55):
    h, w = img.shape[:2]

    y0 = int((1.0 - frac) * 0.5 * h)
    y1 = int(h - y0)

    x0 = int((1.0 - frac) * 0.5 * w)
    x1 = int(w - x0)

    return img[y0:y1, x0:x1]


def classify_floor_color_opencv(tile_bgr):
    """
    Classifies only the fixed floor colors:
        P = purple
        Y = yellow
        B = blue
        M = pink/magenta

    Red and green are reserved for objects, so they are not returned here.
    """
    if tile_bgr is None or tile_bgr.size == 0:
        return "unknown", "?", {"reason": "empty_roi"}

    roi = center_crop(tile_bgr, 0.55)

    if roi.size == 0:
        return "unknown", "?", {"reason": "bad_center_crop"}

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
        return "unknown", "?", {
            "reason": "too_little_colored_area",
            "valid_ratio": round(valid_ratio, 4)
        }

    # Red and green are object colors, so ignore them for floor classification.
    red_mask = (((H <= 10) | (H >= 170)) & valid)
    green_mask = ((H >= 40) & (H <= 90) & valid)

    yellow_mask = ((H >= 18) & (H <= 38) & valid)
    blue_mask = ((H >= 95) & (H <= 135) & valid)
    purple_pink_mask = ((H >= 136) & (H <= 169) & valid)

    ratios = {
        "yellow": float(np.count_nonzero(yellow_mask)) / float(valid.size),
        "blue": float(np.count_nonzero(blue_mask)) / float(valid.size),
        "purple_pink": float(np.count_nonzero(purple_pink_mask)) / float(valid.size),
        "red_reserved_object": float(np.count_nonzero(red_mask)) / float(valid.size),
        "green_reserved_object": float(np.count_nonzero(green_mask)) / float(valid.size),
    }

    h_mean = float(np.mean(H[valid]))
    s_mean = float(np.mean(S[valid]))
    v_mean = float(np.mean(V[valid]))
    a_mean = float(np.mean(A[valid]))
    lb_mean = float(np.mean(LB[valid]))

    best_basic = max(["yellow", "blue", "purple_pink"], key=lambda k: ratios[k])
    best_ratio = ratios[best_basic]

    label = "unknown"
    ch = "?"

    if best_ratio >= 0.15:
        if best_basic == "yellow":
            label, ch = "yellow", "Y"

        elif best_basic == "blue":
            label, ch = "blue", "B"

        elif best_basic == "purple_pink":
            # Pink/magenta is usually brighter/warmer than purple.
            # You may tune these thresholds after seeing debug metrics.
            if v_mean >= 125 or lb_mean >= 145:
                label, ch = "pink", "M"
            else:
                label, ch = "purple", "P"

    # fallback for pink/purple if hue mask was weak
    if label == "unknown":
        if a_mean >= 145 and lb_mean >= 135 and v_mean >= 105:
            label, ch = "pink", "M"
        elif a_mean >= 140 and lb_mean < 140:
            label, ch = "purple", "P"

    metrics = {
        "valid_ratio": round(valid_ratio, 4),
        "h_mean": round(h_mean, 2),
        "s_mean": round(s_mean, 2),
        "v_mean": round(v_mean, 2),
        "lab_a_mean": round(a_mean, 2),
        "lab_b_mean": round(lb_mean, 2),
        "ratios": {k: round(v, 4) for k, v in ratios.items()}
    }

    return label, ch, metrics


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    final_grid = {
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

    detailed = {}

    for heading in HEADINGS:
        path = os.path.join(SCAN_DIR, f"{heading}.jpg")

        if not os.path.exists(path):
            print(f"ERROR: Missing image: {path}")
            return

        img = cv2.imread(path)

        if img is None:
            print(f"ERROR: Could not read image: {path}")
            return

        slots = get_three_slot_rois(img)

        if len(slots) != 3:
            print(f"ERROR: Could not build 3 slots for heading: {heading}")
            return

        heading_info = []
        print(f"\nHeading: {heading}")

        for i, tile in enumerate(slots):
            dbg_name = os.path.join(DEBUG_DIR, f"{heading}_slot{i}.jpg")
            cv2.imwrite(dbg_name, tile)

            label, ch, metrics = classify_floor_color_opencv(tile)

            pos = HEADING_TO_POSITIONS[heading][i]
            final_grid[pos] = ch

            print(f"  slot {i}: floor_label={label}, char={ch}, saved={dbg_name}")
            print(f"    metrics: {metrics}")

            heading_info.append({
                "slot_index": i,
                "pos": [pos[0], pos[1]],
                "label": label,
                "char": ch,
                "debug_crop": dbg_name,
                "metrics": metrics
            })

        detailed[heading] = heading_info

    print("\nFinal local 3x3 FLOOR color matrix:")
    pretty_print_matrix(final_grid)

    out = {
        "center": [0, 0],
        "agent": "A",
        "grid_letters": {
            f"{c},{r}": final_grid[(c, r)]
            for (c, r) in final_grid
        },
        "per_heading": detailed
    }

    json_path = os.path.join(RESULTS_DIR, "color_results.json")
    txt_path = os.path.join(RESULTS_DIR, "local_color_3x3.txt")

    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    save_matrix_txt(txt_path, final_grid)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved debug crops in: {DEBUG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
