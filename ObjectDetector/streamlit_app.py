import streamlit as st
import cv2
import numpy as np
import os

# Fix for MSMF error
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# Set page config
st.set_page_config(page_title="Object Detection", layout="wide")
st.title("Real-Time Object Detection")

# Load model and classes
@st.cache_resource
def load_model():
    thres = 0.45
    classNames = []
    with open('coco.names', 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    return net, classNames, thres

net, classNames, thres = load_model()

# Camera selection
camera_index = st.selectbox("Select Camera", options=[0, 1], index=0)

# Initialize with DirectShow
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    st.error(f"Could not open camera {camera_index}")
    st.stop()

st.success(f"Camera {camera_index} opened successfully")
FRAME_WINDOW = st.image([])
stop_button = st.button("Stop Detection")

try:
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Frame read failed - trying to reconnect...")
            cap.release()
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            continue
            
        # Convert color for processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Object detection
        classIds, confs, bbox = net.detect(frame, confThreshold=thres)
        
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(frame, box, color=(0,255,0), thickness=2)
                cv2.putText(frame, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, str(round(confidence*100,2)), (box[0]+200, box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        
        # Display the frame
        FRAME_WINDOW.image(frame)
        
finally:
    cap.release()
    st.write("Camera released")