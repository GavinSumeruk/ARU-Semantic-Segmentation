import numpy as np
import cv2

# one colour per class slot -- cycles if more than 8 objects
COLOURS = [
    (255,  80,  80),   # red
    ( 80, 200,  80),   # green
    ( 80,  80, 255),   # blue
    (255, 220,  50),   # yellow
    ( 50, 220, 220),   # cyan
    (220,  50, 220),   # magenta
    (255, 160,  50),   # orange
    (160,  50, 255),   # purple
]


def draw_masks(image, pipeline_outputs):
    """
    Paint coloured masks onto an image and add text labels.

    Args:
        image:            numpy array (H, W, 3) -- the original frame
        pipeline_outputs: list of (class_name, confidence, mask)
                          from run_pipeline()

    Returns:
        annotated image as a numpy array (H, W, 3)
    """
    overlay = image.copy()

    for i, (class_name, conf, mask) in enumerate(pipeline_outputs):
        colour = COLOURS[i % len(COLOURS)]

        # paint the mask area with a semi-transparent colour
        coloured = np.zeros_like(image)
        coloured[mask > 0] = colour
        overlay = cv2.addWeighted(overlay, 1.0, coloured, 0.45, 0)

        # draw the mask outline
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, colour, 2)

        # place a text label above the mask
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            x_label = int(xs.mean())
            y_label = int(ys.min()) - 12
            y_label = max(y_label, 15)   # don't go off the top edge
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                overlay, label,
                (x_label, y_label),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, colour, 2, cv2.LINE_AA
            )

    return overlay


# quick test -- runs draw_masks with a fake rectangular mask
if __name__ == "__main__":
    import urllib.request
    urllib.request.urlretrieve(
        "https://ultralytics.com/images/bus.jpg", "bus.jpg"
    )

    image = cv2.imread("bus.jpg")
    h, w  = image.shape[:2]

    # make a fake mask covering the centre of the image
    fake_mask = np.zeros((h, w), dtype=bool)
    fake_mask[100:400, 50:700] = True
    fake_outputs = [("bus", 0.94, fake_mask)]

    result = draw_masks(image, fake_outputs)
    cv2.imwrite("draw_test.jpg", result)
    print("saved draw_test.jpg -- open it to check the overlay looks right")