# main.py
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import deque
import time
from datetime import datetime

# ------------------- Page config -------------------
st.set_page_config(page_title="AI Traffic Dashboard", layout="wide")

# ------------------- Session state -------------------
if "running" not in st.session_state:
    st.session_state.running = True
if "manual_overrides" not in st.session_state:
    st.session_state.manual_overrides = {}
if "auto_mode" not in st.session_state:
    st.session_state.auto_mode = True
if "ambulance_cycle" not in st.session_state:
    st.session_state.ambulance_cycle = 0
if "flash_toggle" not in st.session_state:
    st.session_state.flash_toggle = False

# ------------------- Model -------------------
model = YOLO("yolov8n.pt")

# ------------------- Sources -------------------
mode = st.sidebar.radio("📡 Feed Mode", ("Demo Videos", "Public Live Streams"))
if mode == "Public Live Streams":
    video_paths = [
        "http://wzmedia.dot.ca.gov/D4/S880_at_Paseo_Grande_OC.stream/playlist.m3u8",
        "http://wzmedia.dot.ca.gov/D4/S238_NOF_Ashland_UC.stream/playlist.m3u8",
        "lane3.mp4",
        "lane4.mp4",
    ]
else:
    video_paths = ["laneA.mp4", "laneB.mp4", "laneC.mp4", "laneD.mp4"]

caps = [cv2.VideoCapture(v) for v in video_paths]

# ------------------- Parameters -------------------
min_green, max_green, max_count, history_len = 5, 180, 100, 50
num_lanes = len(video_paths)
history = [deque(maxlen=history_len) for _ in range(num_lanes)]

# ------------------- Helpers -------------------
def resize_frame(frame, width=480):
    if frame is None:
        return np.zeros((int(width*0.6), width, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    new_h = int(width * (h / w))
    return cv2.resize(frame, (width, new_h))

def draw_boxes(frame, results):
    """Draw bounding boxes without text labels."""
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        color = (0, 255, 0) if cls_id in [2, 3, 5, 7] else (0, 0, 255)  # Red for ambulance
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame

def plot_charts(lane_counts, history):
    fig, axes = plt.subplots(2, 1, figsize=(5, 6))
    lanes = [f"Lane {i+1}" for i in range(len(lane_counts))]
    axes[0].bar(lanes, lane_counts, color=["#0077b6", "#00b4d8", "#90e0ef", "#caf0f8"][:len(lane_counts)])
    axes[0].set_ylim(0, max_count)
    axes[0].set_title("Live Vehicle Count")
    axes[1].set_title(f"Traffic Trend (last {history_len} frames)")
    for i in range(len(history)):
        axes[1].plot(list(history[i]), label=f"Lane {i+1}")
    if len(history) > 0:
        axes[1].legend()
    fig.tight_layout()
    return fig

def apply_overrides_with_ambulance(lane_counts, ambulance_counts):
    now = time.time()
    expired = [lane for lane, info in st.session_state.manual_overrides.items()
               if info.get("expires_at") and now >= info["expires_at"]]
    for lane in expired:
        del st.session_state.manual_overrides[lane]

    lane_times = [min_green + (c / max_count) * (max_green - min_green) for c in lane_counts]
    lane_times = [min(t, max_green) for t in lane_times]

    manual_active = []
    for lane_idx, info in st.session_state.manual_overrides.items():
        if 0 <= lane_idx < num_lanes:
            manual_active.append(lane_idx)
            dur = info.get("duration")
            if dur and 0 < dur <= 3600:
                lane_times[lane_idx] = dur

    ambulance_lanes = [i for i, count in enumerate(ambulance_counts) if count > 0]
    if ambulance_lanes:
        lane_idx = ambulance_lanes[st.session_state.ambulance_cycle % len(ambulance_lanes)]
        lane_times[lane_idx] = max_green
        st.session_state.ambulance_cycle += 1
        return [lane_idx], lane_times

    if st.session_state.auto_mode and not st.session_state.manual_overrides:
        active_lanes = list(np.argsort(lane_counts)[-2:]) if any(lane_counts) else [0, 1]
    else:
        active_lanes = manual_active

    return active_lanes, lane_times

# ------------------- Header -------------------
st.title("🚦 AI Traffic Monitoring Dashboard")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📍 Location", "Paseo Grande / Ashland UC")
with col2:
    st.metric("🛣 Lanes Monitored", str(num_lanes))
with col3:
    st.metric("⏰ Time", datetime.now().strftime("%H:%M:%S"))

# ------------------- Sidebar -------------------
with st.sidebar:
    st.header("⚙️ Controls")
    st.session_state.auto_mode = st.checkbox("Enable Auto Mode", value=True)

    st.subheader("Manual Overrides")
    lane_to_control = st.selectbox("Select Lane", [f"Lane {i+1}" for i in range(num_lanes)], index=0)
    lane_idx = int(lane_to_control.split()[-1]) - 1
    manual_action = st.radio("Action", ("Set GREEN", "Set RED"))
    manual_duration = st.number_input("Duration (seconds)", min_value=0, max_value=3600, value=20, step=5)

    colA, colB = st.columns(2)
    if colA.button("Apply Override"):
        info = {
            "state": "GREEN" if manual_action == "Set GREEN" else "RED",
            "duration": manual_duration if manual_duration > 0 else None,
            "expires_at": (time.time() + manual_duration) if manual_duration > 0 else None,
            "applied_at": time.time(),
            "user": "Controller"
        }
        st.session_state.manual_overrides[lane_idx] = info
        st.success(f"Applied {info['state']} to Lane {lane_idx+1}")

    if colB.button("Release Override"):
        if lane_idx in st.session_state.manual_overrides:
            del st.session_state.manual_overrides[lane_idx]
            st.success(f"Released override on Lane {lane_idx+1}")

    if st.button("Clear All Overrides"):
        st.session_state.manual_overrides.clear()
        st.success("Cleared all manual overrides")

    st.write("### Active Overrides")
    if st.session_state.manual_overrides:
        for lane, info in st.session_state.manual_overrides.items():
            exp = datetime.fromtimestamp(info["expires_at"]).strftime("%H:%M:%S") if info.get("expires_at") else "Indefinite"
            st.write(f"Lane {lane+1}: {info['state']} (till {exp})")
    else:
        st.write("No active overrides")

    st.subheader("▶ Control Monitoring")
    if st.button("Start Monitoring"):
        st.session_state.running = True
    if st.button("Pause Monitoring"):
        st.session_state.running = False

# ------------------- Dashboard Placeholders -------------------
stats_col1, stats_col2, stats_col3 = st.columns(3)
total_placeholder = stats_col1.metric("🚗 Total Vehicles", "0")
active_placeholder = stats_col2.metric("✅ Active Lanes", "0")
congestion_placeholder = stats_col3.metric("⚠️ Congestion Level", "Normal")

alert_placeholder = st.empty()
video_cols = st.columns(2)
lane_video_placeholders = [video_cols[i % 2].empty() for i in range(num_lanes)]
lane_info_placeholders = [video_cols[i % 2].empty() for i in range(num_lanes)]
chart_placeholder = st.empty()

# ------------------- Main loop -------------------
while st.session_state.running:
    frames, lane_counts, ambulance_counts = [], [], []

    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        vehicle_count, amb_count = 0, 0
        if ret and frame is not None:
            results = model(frame, verbose=False)
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id in [2, 3, 5, 7]:
                    vehicle_count += 1
                elif cls_id == 1:  # Example ambulance ID
                    amb_count += 1
            frame = draw_boxes(frame, results)
        else:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)

        lane_counts.append(vehicle_count)
        ambulance_counts.append(amb_count)
        frames.append(resize_frame(frame))

    active_lanes, lane_times = apply_overrides_with_ambulance(lane_counts, ambulance_counts)

    # 🚨 Ambulance alert
    ambulance_present = any(a > 0 for a in ambulance_counts)
    st.session_state.flash_toggle = not st.session_state.flash_toggle if ambulance_present else False
    alert_color = "#ff4d4d" if st.session_state.flash_toggle else "#f0f0f0"
    alert_placeholder.markdown(
        f"<div style='background-color:{alert_color}; text-align:center; font-size:22px; font-weight:bold; padding:8px; border-radius:10px;'>🚨 Ambulance Detected 🚨</div>"
        if ambulance_present else "",
        unsafe_allow_html=True
    )

    # Show videos and lane info cards
    for i, frame in enumerate(frames):
        lane_video_placeholders[i].image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        info_text = f"""
        <div style='
            text-align:center; 
            padding:10px; 
            margin:10px 0; 
            border-radius:10px; 
            background-color:#f0f4f8; 
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2); 
            font-size:16px;
            color:#0077b6;
            font-weight:bold;
        '>
            Lane {i+1} 🚦 <br>
            Vehicles: {lane_counts[i]} | Green Time: {lane_times[i]:.1f}s
            {" | 🟠 MANUAL" if i in st.session_state.manual_overrides else ""}
            {" | 🚨 AMBULANCE" if ambulance_counts[i] > 0 else ""}
        </div>
        """
        lane_info_placeholders[i].markdown(info_text, unsafe_allow_html=True)
        history[i].append(lane_counts[i])

    # Update metrics
    total = sum(lane_counts)
    active_count = len(active_lanes)
    congestion = "Severe" if total > 200 else "High" if total > 100 else "Normal"
    total_placeholder.metric("🚗 Total Vehicles", str(total))
    active_placeholder.metric("✅ Active Lanes", str(active_count))
    congestion_placeholder.metric("⚠️ Congestion Level", congestion)

    # Charts
    fig = plot_charts(lane_counts, history)
    chart_placeholder.pyplot(fig)
    plt.close(fig)

    time.sleep(0.1)

st.warning("Monitoring paused. ▶ Start Monitoring from sidebar to resume.")
