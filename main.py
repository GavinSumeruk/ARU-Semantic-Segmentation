import cv2
import numpy as np
from ultralytics import YOLO
from pipeline import run_pipeline
from draw_masks import draw_masks


def load_models():
    """
    Load YOLO and NanoSAM once at startup.
    Swap the FakeSegmenter for the real NanoSAM predictor
    once engines are built on the Orin.
    """
    detector = YOLO("yolo11n.pt")

    # ---- swap this block for real NanoSAM once engines are built ----
    class FakeSegmenter:
        def set_image(self, img):
            self.h, self.w = img.shape[:2]
        def predict(self, points, labels):
            mask = np.zeros((self.h, self.w), dtype=bool)
            x1, y1 = int(points[0][0]), int(points[0][1])
            x2, y2 = int(points[1][0]), int(points[1][1])
            mask[y1:y2, x1:x2] = True
            return mask, None, None

    segmenter = FakeSegmenter()
    # ---- end of placeholder block ----

    # real NanoSAM will look like this:
    # from nanosam.utils.predictor import Predictor
    # segmenter = Predictor(
    #     "data/resnet18_image_encoder.engine",
    #     "data/mobile_sam_mask_decoder.engine"
    # )

    return detector, segmenter


def run_on_image(image_path, detector, segmenter):
    """Run the full pipeline on one image file and save the result."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"could not load image: {image_path}")
        return

    results = run_pipeline(image, detector, segmenter)
    print(f"found {len(results)} objects:")
    for name, conf, mask in results:
        print(f"  {name}  conf={conf:.2f}  "
              f"mask covers {mask.sum()} pixels")

    annotated = draw_masks(image, results)
    out_path = "output.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"saved annotated image to {out_path}")


def run_on_webcam(detector, segmenter):
    """Run the pipeline live on a webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("could not open webcam")
        return

    print("running live -- press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results  = run_pipeline(frame, detector, segmenter)
        annotated = draw_masks(frame, results)

        cv2.imshow("pipeline", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    detector, segmenter = load_models()

    if len(sys.argv) > 1:
        # python3 main.py my_image.jpg
        run_on_image(sys.argv[1], detector, segmenter)
    else:
        # python3 main.py  (no argument = webcam)
        run_on_image("bus.jpg", detector, segmenter)