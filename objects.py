import os
import json
import cv2
import numpy as np

# =========================================================
# CONFIG
# =========================================================

SCAN_DIR = "scan_images"
DEBUG_DIR = "debug_objects"
RESULTS_DIR = "results"

HEADINGS = ["front", "right", "back", "left"]

# Same ROI as color detector
ROI_TOP_FRAC = 0.55
ROI_BOT_FRAC = 0.95

SLOT_PAD_X_FRAC = 0.03
SLOT_PAD_Y_FRAC = 0.06

# Object colors:
#   Green box = Target
#   Red box   = Obstacle
GREEN_S_MIN = 70
GREEN_V_MIN = 70

RED_S_MIN = 80
RED_V_MIN = 70

# Detection thresholds
MIN_OBJECT_RATIO = 0.035
MIN_CONTOUR_AREA_FRAC = 0.015
MAX_CONTOUR_AREA_FRAC = 0.75

MIN_OBJECT_W_FRAC = 0.10
MIN_OBJECT_H_FRAC = 0.10

# If object is roughly inside a tile slot, this helps reject noise
MAX_CENTER_OFFSET = 0.75

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


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def get_largest_valid_blob(mask, slot_shape):
    h, w = slot_shape[:2]
    slot_area = h * w

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

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

        if w_frac < MIN_OBJECT_W_FRAC or h_frac < MIN_OBJECT_H_FRAC:
            continue

        cx = x + bw / 2.0
        cy = y + bh / 2.0

        center_dx = abs(cx - w / 2.0) / max(1.0, w / 2.0)
        center_dy = abs(cy - h / 2.0) / max(1.0, h / 2.0)

        if center_dx > MAX_CENTER_OFFSET or center_dy > MAX_CENTER_OFFSET:
            continue

        if area > best_area:
            best_area = area
            best = {
                "x": int(x),
                "y": int(y),
                "w": int(bw),
                "h": int(bh),
                "area": float(area),
                "area_frac": float(area_frac),
                "w_frac": float(w_frac),
                "h_frac": float(h_frac),
                "center_dx": float(center_dx),
                "center_dy": float(center_dy),
            }

    return best


def detect_one_object_slot(slot_bgr):
    """
    Return:
        T = green target object
        O = red obstacle object
        E = empty
        ? = uncertain

    Red/green are reserved for objects only.
    """
    if slot_bgr is None or slot_bgr.size == 0:
        return "?", {"reason": "empty_slot"}

    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)

    # Red has two hue ranges in HSV
    lower_red1 = np.array([0, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([170, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red1, red2)

    # Green target
    lower_green = np.array([40, GREEN_S_MIN, GREEN_V_MIN], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_mask = clean_mask(red_mask)
    green_mask = clean_mask(green_mask)

    red_ratio = float(np.count_nonzero(red_mask)) / float(red_mask.size)
    green_ratio = float(np.count_nonzero(green_mask)) / float(green_mask.size)

    red_blob = get_largest_valid_blob(red_mask, slot_bgr.shape)
    green_blob = get_largest_valid_blob(green_mask, slot_bgr.shape)

    metrics = {
        "red_ratio": round(red_ratio, 4),
        "green_ratio": round(green_ratio, 4),
        "red_blob": red_blob,
        "green_blob": green_blob,
    }

    red_detected = red_ratio >= MIN_OBJECT_RATIO and red_blob is not None
    green_detected = green_ratio >= MIN_OBJECT_RATIO and green_blob is not None

    if red_detected and green_detected:
        # Pick stronger object.
        if red_ratio > green_ratio:
            return "O", metrics
        else:
            return "T", metrics

    if red_detected:
        return "O", metrics

    if green_detected:
        return "T", metrics

    return "E", metrics


def save_debug_masks(slot_bgr, heading, slot_index):
    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([170, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red1, red2)
    red_mask = clean_mask(red_mask)

    lower_green = np.array([40, GREEN_S_MIN, GREEN_V_MIN], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_mask = clean_mask(green_mask)

    cv2.imwrite(os.path.join(DEBUG_DIR, f"{heading}_slot{slot_index}_red_mask.jpg"), red_mask)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{heading}_slot{slot_index}_green_mask.jpg"), green_mask)


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

            save_debug_masks(tile, heading, i)

            obj_char, metrics = detect_one_object_slot(tile)

            pos = HEADING_TO_POSITIONS[heading][i]
            final_grid[pos] = obj_char

            print(f"  slot {i}: object={obj_char}, saved={dbg_name}")
            print(f"    metrics: {metrics}")

            heading_info.append({
                "slot_index": i,
                "pos": [pos[0], pos[1]],
                "object": obj_char,
                "debug_crop": dbg_name,
                "metrics": metrics
            })

        detailed[heading] = heading_info

    print("\nFinal local 3x3 OBJECT matrix:")
    pretty_print_matrix(final_grid)

    out = {
        "center": [0, 0],
        "agent": "A",
        "legend": {
            "T": "green target object",
            "O": "red obstacle object",
            "E": "empty",
            "?": "uncertain",
            "A": "agent"
        },
        "grid_objects": {
            f"{c},{r}": final_grid[(c, r)]
            for (c, r) in final_grid
        },
        "per_heading": detailed
    }

    json_path = os.path.join(RESULTS_DIR, "object_results.json")
    txt_path = os.path.join(RESULTS_DIR, "local_object_3x3.txt")

    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    save_matrix_txt(txt_path, final_grid)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved debug crops and masks in: {DEBUG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
