
import sys
import time
from pathlib import Path

import cv2
import serial
from ultralytics import YOLO

# -------- CHANGE THESE SETTINGS --------
MODEL_PATH = Path(__file__).parent / "best.pt"
RESULTS_DIR = Path(__file__).parent / "results"

SERIAL_PORT = "COM3"   # Change to your Arduino COM port
BAUD_RATE = 9600
CONFIDENCE = 0.40

CLASS_MAP = {
    "neutrophil": "N",
    "neutrophils": "N",
    "lymphocyte": "L",
    "lymphocytes": "L",
    "monocyte": "M",
    "monocytes": "M",
}
# ---------------------------------------


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        return

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        print("Image not found:", image_path)
        return

    if not MODEL_PATH.is_file():
        print("YOLO model not found:", MODEL_PATH)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        print("Could not read image:", image_path)
        return

    model = YOLO(str(MODEL_PATH))
    result = model.predict(
        image, conf=CONFIDENCE, verbose=False
    )[0]

    counts = {"N": 0, "L": 0, "M": 0}
    annotated = image.copy()

    for box in result.boxes:
        class_id = int(box.cls.item())
        name = str(result.names[class_id]).lower().strip()
        category = CLASS_MAP.get(name)

        if category is None:
            print("Unmapped model class:", name)
            continue

        counts[category] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf.item())

        cv2.rectangle(
            annotated, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )
        cv2.putText(
            annotated, f"{name} {confidence:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2
        )

    n, l, m = counts["N"], counts["L"], counts["M"]

    lines = [
        f"Neutrophils: {n}",
        f"Lymphocytes: {l}",
        f"Monocytes: {m}",
        f"NLR: {n / l:.3f}" if l else "NLR: Undefined (L=0)",
        f"MLR: {m / l:.3f}" if l else "MLR: Undefined (L=0)",
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            annotated, line, (10, 25 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255), 2
        )

    output_path = RESULTS_DIR / f"{image_path.stem}_result.jpg"
    cv2.imwrite(str(output_path), annotated)

    print("\nCell analysis")
    print(*lines, sep="\n")
    print("Annotated image:", output_path)

    # Show the image on the computer.
    cv2.namedWindow("YOLO Optical Image", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLO Optical Image", annotated)

    # If L=0, ratios are undefined; do not transmit them.
    if l > 0:
        message = f"{n},{l},{m},{n/l:.3f},{m/l:.3f}\n"

        try:
            # This works only when Java has released the same COM port.
            with serial.Serial(
                SERIAL_PORT, BAUD_RATE, timeout=2
            ) as arduino:
                time.sleep(2)  # Arduino may reset on serial connection.
                arduino.write(message.encode("ascii"))
                arduino.flush()
                print("Sent to Arduino:", message.strip())

                reply = arduino.readline().decode(
                    "utf-8", errors="replace"
                ).strip()
                if reply:
                    print("Arduino response:", reply)

        except serial.SerialException as exc:
            print("Arduino serial error:", exc)
            print("Check COM port and ensure Java released it.")
    else:
        print("Ratios not sent: lymphocyte count is zero.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
