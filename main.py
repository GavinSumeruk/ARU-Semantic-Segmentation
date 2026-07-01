import cv2
import numpy as np
from ultralytics import YOLO
from pipeline import run_pipeline
from draw_masks import draw_masks


class UltralyticsSAMSegmenter:
    """
    Wraps Ultralytics MobileSAM to match the NanoSAM predictor interface.
    Provides set_image() and predict() so pipeline.py works unchanged.
    When NanoSAM engines are built, swap this class for the real Predictor.
    """
    def __init__(self):
        from ultralytics import SAM
        print("Loading MobileSAM...")
        self.model = SAM("mobile_sam.pt")
        self.image = None
        print("MobileSAM loaded.")

    def set_image(self, image):
        """Store the current frame. Called once per frame before predict()."""
        self.image = image

    def predict(self, points, labels):
        """
        Segment the object at the given prompt.
        points: numpy array [[x1,y1],[x2,y2]] -- box corners
        labels: numpy array [2,3]             -- NanoSAM box label convention
        Returns: (mask, None, None) to match NanoSAM's return signature.
        """
        if self.image is None:
            raise RuntimeError("Call set_image() before predict()")

        # convert NanoSAM box-corner format to a flat [x1,y1,x2,y2] bbox
        x1, y1 = float(points[0][0]), float(points[0][1])
        x2, y2 = float(points[1][0]), float(points[1][1])

        results = self.model(
            self.image,
            bboxes=[[x1, y1, x2, y2]],
            verbose=False
        )

        # if no mask came back return a blank mask rather than crashing
        if results[0].masks is None:
            h, w = self.image.shape[:2]
            return np.zeros((h, w), dtype=bool), None, None

        mask = results[0].masks.data[0].cpu().numpy().astype(bool)
        return mask, None, None


def load_models():
    """
    Load YOLO and the segmenter once at startup.
    To swap to real NanoSAM later, replace UltralyticsSAMSegmenter()
    with the three commented lines below.
    """
    detector = YOLO("yolo11n.pt")
    segmenter = UltralyticsSAMSegmenter()

    # ---- swap to real NanoSAM once engines are built ----
    # from nanosam.utils.predictor import Predictor
    # segmenter = Predictor(
    #     "/home/user/nanosam/data/resnet18_image_encoder.engine",
    #     "/home/user/nanosam/data/mobile_sam_mask_decoder.engine"
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
        print(f"  {name}  conf={conf:.2f}  mask covers {mask.sum()} pixels")

    annotated = draw_masks(image, results)
    out_path = "output.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"saved annotated image to {out_path}")


def run_on_webcam(detector, segmenter):
    """Run the pipeline live on a webcam or camera feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("could not open camera")
        return

    print("running live -- press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results   = run_pipeline(frame, detector, segmenter)
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
        # python3 main.py  (no argument = bus.jpg default)
        run_on_image("bus.jpg", detector, segmenter)