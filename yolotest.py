from ultralytics import YOLO

yolo = YOLO("./yolo11n.pt")
# source=0是摄像头,screen是检测屏幕
results = yolo(source="./ultralytics/assets/bus.jpg", save=False)
# results = yolo(source=0,show=False)

# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box

