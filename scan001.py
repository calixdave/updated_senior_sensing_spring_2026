import cv2
import os
import time

# =========================================================
# Simple Pi-friendly scan capture
# Saves 4 images only: front, right, back, left
#
# Keys:
#   c = capture current heading
#   q = quit
# =========================================================

SAVE_DIR = "scan_images"
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

HEADINGS = ["front", "right", "back", "left"]

# Lower ROI to focus more on the first row of tiles in front of the robot
ROI_TOP_FRAC = 0.55
ROI_BOT_FRAC = 0.95

SLOT_PAD_X_FRAC = 0.03
SLOT_PAD_Y_FRAC = 0.06

CLEAR_OLD_IMAGES_ON_START = True


def put_text(img, text, y, scale=0.7, thickness=2):
    cv2.putText(
        img,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )


def draw_slot_guides(img):
    h, w = img.shape[:2]
    y0 = int(ROI_TOP_FRAC * h)
    y1 = int(ROI_BOT_FRAC * h)

    if y1 <= y0:
        return img

    out = img.copy()
    cv2.rectangle(out, (0, y0), (w - 1, y1), (255, 255, 0), 1)

    band_h = y1 - y0

    for i in range(3):
        sx0 = int(i * w / 3)
        sx1 = int((i + 1) * w / 3)

        pad_x = int(SLOT_PAD_X_FRAC * (sx1 - sx0))
        pad_y = int(SLOT_PAD_Y_FRAC * band_h)

        cx0 = max(0, sx0 + pad_x)
        cx1 = min(w, sx1 - pad_x)
        cy0 = max(0, y0 + pad_y)
        cy1 = min(h, y1 - pad_y)

        cv2.rectangle(out, (cx0, cy0), (cx1, cy1), (0, 255, 0), 2)

        cv2.putText(
            out,
            f"slot {i}",
            (cx0 + 5, cy0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return out


def clear_old_scan_images():
    os.makedirs(SAVE_DIR, exist_ok=True)

    for heading in HEADINGS:
        path = os.path.join(SAVE_DIR, f"{heading}.jpg")
        if os.path.exists(path):
            os.remove(path)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if CLEAR_OLD_IMAGES_ON_START:
        clear_old_scan_images()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("Camera opened.")
    print("Instructions:")
    print("  Press 'c' to capture current heading")
    print("  Press 'q' to quit")
    print("Capture order: front -> right -> back -> left")

    idx = 0
    last_capture_msg = ""

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("ERROR: Failed to read frame from camera.")
            break

        display = draw_slot_guides(frame)

        if idx < len(HEADINGS):
            current_heading = HEADINGS[idx]
            put_text(display, f"Current heading: {current_heading}", 30, 0.9, 2)
            put_text(display, "Press 'c' to capture this view", 65)
            put_text(display, "Rotate robot manually before each capture", 95)
            put_text(display, "Green boxes = front-row tile slots", 125)
            put_text(display, "Press 'q' to quit", 155)
        else:
            put_text(display, "All 4 captures completed.", 30, 0.9, 2)
            put_text(display, "Press 'q' to quit", 65)

        if last_capture_msg:
            put_text(display, last_capture_msg, FRAME_HEIGHT - 20, 0.65, 2)

        cv2.imshow("capture_scan", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quit requested.")
            break

        elif key == ord("c"):
            if idx >= len(HEADINGS):
                print("All headings already captured.")
                last_capture_msg = "All headings already captured"
                continue

            heading = HEADINGS[idx]
            filename = os.path.join(SAVE_DIR, f"{heading}.jpg")

            ok = cv2.imwrite(filename, frame)

            if ok:
                print(f"Saved: {filename}")
                last_capture_msg = f"Saved {heading}.jpg"
                idx += 1
                time.sleep(0.4)
            else:
                print(f"ERROR: Failed to save {filename}")
                last_capture_msg = f"Failed to save {heading}.jpg"

    cap.release()
    cv2.destroyAllWindows()
    print("Program ended.")


if __name__ == "__main__":
    main()
