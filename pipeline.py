import numpy as np
from glue import box_to_prompt


def run_pipeline(image, detector, segmenter):
    """
    Full detect-then-segment pipeline.

    Args:
        image:     numpy array (H, W, 3) -- one camera frame
        detector:  loaded YOLO model
        segmenter: loaded NanoSAM predictor

    Returns:
        list of (class_name, confidence, mask) tuples
        -- one entry per detected object
        -- mask is a boolean array the same H x W as the image
    """
    # ---- 1. DETECT ----
    # run YOLO on the image, get back boxes, class IDs, confidences
    results = detector(image, verbose=False)
    boxes   = results[0].boxes.xyxy.cpu().numpy()   # shape (N, 4)
    classes = results[0].boxes.cls.cpu().numpy()     # shape (N,)
    confs   = results[0].boxes.conf.cpu().numpy()    # shape (N,)
    names   = results[0].names                       # {id: "name"}

    # if nothing was detected, return empty list -- don't crash
    if len(boxes) == 0:
        return []

    # ---- 2. ENCODE THE IMAGE ONCE ----
    # this is the expensive step -- do it once, not inside the loop
    segmenter.set_image(image)

    outputs = []

    # ---- 3. SEGMENT each detected box ----
    for box, cls, conf in zip(boxes, classes, confs):

        # convert the YOLO box to the prompt format NanoSAM wants
        points, labels = box_to_prompt(box)

        # ask NanoSAM to segment whatever is in this box
        mask, _, _ = segmenter.predict(points, labels)

        class_name = names[int(cls)]
        outputs.append((class_name, float(conf), mask))

    return outputs


# quick test using a stand-in segmenter (no NanoSAM needed)
if __name__ == "__main__":
    import cv2
    from ultralytics import YOLO

    # --- stand-in segmenter so we can test without NanoSAM ---
    class FakeSegmenter:
        """Returns a blank mask the same size as the image."""
        def set_image(self, img):
            self.h, self.w = img.shape[:2]
        def predict(self, points, labels):
            mask = np.zeros((self.h, self.w), dtype=bool)
            # fill the box area so we can see something
            x1, y1 = int(points[0][0]), int(points[0][1])
            x2, y2 = int(points[1][0]), int(points[1][1])
            mask[y1:y2, x1:x2] = True
            return mask, None, None

    # load a real image and run
    import urllib.request
    urllib.request.urlretrieve(
        "https://ultralytics.com/images/bus.jpg", "bus.jpg"
    )

    detector  = YOLO("yolo11n.pt")
    segmenter = FakeSegmenter()

    image = cv2.imread("bus.jpg")
    results = run_pipeline(image, detector, segmenter)

    print(f"found {len(results)} objects:")
    for name, conf, mask in results:
        print(f"  {name}  conf={conf:.2f}  "
              f"mask covers {mask.sum()} pixels")