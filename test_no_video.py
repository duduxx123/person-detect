import cv2
from ultralytics import YOLO

# 加载 YOLO 模型
yolo = YOLO("./yolo11n.pt")

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# --- 新增：视频写入配置 ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20  # 设置帧率
# 定义编码器并创建 VideoWriter 对象，保存为 output.mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))
print("正在录制并检测，按 Ctrl+C 停止并在本地查看 output.mp4")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(source=frame, verbose=False)

        for result in results:
            cls_ids = result.boxes.cls.int().tolist()
            names_dict = result.names
            person_indices = [i for i, cls in enumerate(cls_ids) if names_dict[cls] == "person"]

            if person_indices:
                boxes = result.boxes.xyxy[person_indices]
                confs = result.boxes.conf[person_indices]

                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[j].int().tolist()
                    conf = confs[j].item()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- 修改处：不再使用 cv2.imshow ---
        # 1. 将处理后的帧写入视频文件
        out.write(frame)

        # 2. 在控制台打印一点反馈，证明程序在跑
        if person_indices:
            print(f"检测到 {len(person_indices)} 个人")

except KeyboardInterrupt:
    print("停止录制")

# 释放资源
cap.release()
out.release()  # 记得释放写入器
cv2.destroyAllWindows()