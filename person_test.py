import cv2
from ultralytics import YOLO

# 加载 YOLO 模型，这里假设使用的是 "yolo11n.pt" 权重文件
yolo = YOLO("./yolo11n.pt")

# 打开摄像头，source=0 表示默认摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取视频帧")
        break

    # 运行 YOLO 检测，传入当前帧
    # 这里 verbose=False 用于降低日志输出
    results = yolo(source=frame, verbose=False)

    # 遍历检测结果（对于 YOLO 模型通常只有一个结果）
    for result in results:
        # 获取每个检测框的类别索引列表（tensor）
        cls_ids = result.boxes.cls.int().tolist()
        # 获取类别名称映射字典（例如 {0: 'person', 1: 'bicycle', ...}）
        names_dict = result.names

        # 过滤出类别为 "person" 的检测框索引
        person_indices = [i for i, cls in enumerate(cls_ids) if names_dict[cls] == "person"]

        # 如果检测到人，则绘制检测框
        if person_indices:
            # 根据索引提取对应的边界框和置信度
            boxes = result.boxes.xyxy[person_indices]  # 每个 box: [x1, y1, x2, y2]
            confs = result.boxes.conf[person_indices]  # 置信度

            # 遍历所有检测到的人，绘制边界框和置信度文本
            for j in range(len(boxes)):
                # 将边界框转换为整数
                x1, y1, x2, y2 = boxes[j].int().tolist()
                conf = confs[j].item()
                # 绘制绿色边框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 在框上绘制置信度文字
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # 如果没有检测到人，则不绘制任何框

    # 显示当前帧
    cv2.imshow("YOLO Person Detection", frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
