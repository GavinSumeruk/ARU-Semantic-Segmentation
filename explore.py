from ultralytics import YOLO

model = YOLO("yolo11n.pt")
r = model("https://ultralytics.com/images/bus.jpg")

boxes = r[0].boxes
print("number of objects found:", len(boxes))
print("box corners (x1,y1,x2,y2):")
print(boxes.xyxy)
print("class id per box:", boxes.cls)
print("confidence per box:", boxes.conf)
print("class names available:", model.names)