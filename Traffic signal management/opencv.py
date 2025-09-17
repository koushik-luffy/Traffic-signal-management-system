import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Video sources (replace with your actual file paths or RTSP/USB cams)
video_paths = [
    r"laneA.mp4",
    r"laneB.mp4",
    r"laneC.mp4",
    r"laneD.mp4"
]

caps = [cv2.VideoCapture(v) for v in video_paths]

# Check if videos opened
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"❌ Error: Could not open {video_paths[i]}")

# Parameters
min_green = 5
max_green = 180
max_count = 100  # max cars to normalize

# Resize helper
def resize_frame(frame, width=640, height=360):
    return cv2.resize(frame, (width, height))

while True:
    frames = []
    lane_counts = []

    # --- Read and detect vehicles ---
    for lane_id, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)  # placeholder if video ends
        else:
            results = model(frame, verbose=False)
            boxes = results[0].boxes
            vehicle_count = 0

            for box in boxes:
                cls_id = int(box.cls[0])  # class id
                if cls_id in [2, 3, 5, 7]:  # 2=car, 3=motorbike, 5=bus, 7=truck (COCO IDs)
                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            lane_counts.append(vehicle_count)
            cv2.putText(frame, f"Lane {lane_id+1}: {vehicle_count} cars",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

        frames.append(resize_frame(frame))

    # --- Scheduler Logic ---
    lane_times = []
    for count in lane_counts:
        green = min_green + (count / max_count) * (max_green - min_green)
        green = min(green, max_green)
        lane_times.append(green)

    # Get indices of top 2 lanes
    if lane_counts:
        top_lanes = np.argsort(lane_counts)[-2:]  # two lanes with highest vehicle counts
    else:
        top_lanes = [0, 1]

    # Annotate each frame with green times and ACTIVE status
    for i, frame in enumerate(frames):
        if i in top_lanes:
            color = (0, 255, 0)  # green for active lanes
            status_text = "ACTIVE"
        else:
            color = (0, 0, 255)  # red for inactive lanes
            status_text = ""

        cv2.putText(frame, f"Green Time: {lane_times[i]:.1f}s",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if status_text:
            cv2.putText(frame, status_text, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # --- Combine into dashboard ---
    top_row = cv2.hconcat([frames[0], frames[1]])
    bottom_row = cv2.hconcat([frames[2], frames[3]])
    dashboard = cv2.vconcat([top_row, bottom_row])

    cv2.imshow("Traffic Dashboard", dashboard)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
