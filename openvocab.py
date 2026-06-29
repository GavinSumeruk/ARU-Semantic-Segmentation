from ultralytics import YOLOWorld

# load an open-vocabulary model (downloads on first run)
model = YOLOWorld("yolov8s-world.pt")

# YOU define the vocabulary — just give it words
model.set_classes(["potted plant", "green tree", "balcony", "window", "person", "bus"])
print("model classes are now:", model.names)

# run on the same bus image
r = model("https://ultralytics.com/images/bus.jpg", conf=0.01)

# save an annotated copy so you can see the boxes
r[0].save("openvocab_result.jpg")

# print what it found, decoded to names
boxes = r[0].boxes
names = r[0].names
print("number of objects found:", len(boxes))
for cls, conf in zip(boxes.cls, boxes.conf):
    print(f"  {names[int(cls)]}   confidence {float(conf):.2f}")