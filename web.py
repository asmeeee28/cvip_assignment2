import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use other YOLOv8 models

# Callback for processing video frames
def process_frame(frame):
    img = frame.to_ndarray(format="bgr24")     # Convert frame to image
    results = model.track(img, tracker="bytetrack.yaml")  # Run object detection + tracking
    annotated_frame = results[0].plot()          # Plot results on frame
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

st.title("Real-Time Object Detection & Tracking with YOLOv8")
st.write("Live analytics platform: detects and tracks objects in webcam stream.")

# Stream from webcam (real-time)
webrtc_streamer(
    key="object-detect",
    video_frame_callback=process_frame,
    sendback_audio=False,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Optional: Upload video files for analysis instead of live webcam
uploaded_file = st.file_uploader("Or upload video for analysis", type=["mp4", "mov", "avi", "mkv"])
if uploaded_file is not None:
    temp_file = f"temp_{uploaded_file.name}"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.read())
    cap = cv2.VideoCapture(temp_file)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, tracker="bytetrack.yaml")
        annotated_frame = results[0].plot()
        stframe.image(annotated_frame, channels="BGR")
    cap.release()