import numpy as np

def box_to_prompt(box):
    """Turn a YOLO box [x1,y1,x2,y2] into a NanoSAM box-prompt:
       two corner points, labelled 2 (top-left) and 3 (bottom-right)."""
    x1, y1, x2, y2 = box
    points = np.array([[x1, y1], [x2, y2]])
    labels = np.array([2, 3])
    return points, labels


# self-test using a real YOLO box
if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    r = model("https://ultralytics.com/images/bus.jpg")
    first_box = r[0].boxes.xyxy[0].cpu().numpy()
    print("YOLO box:", first_box)
    pts, lbls = box_to_prompt(first_box)
    print("prompt points:", pts)
    print("prompt labels:", lbls)