import streamlit as st
import cv2
import mediapipe as mp
import util as util
import numpy as np
from pynput.mouse import Button, Controller
import random
import time
import import_ipynb
import project


# Initialize pynput mouse controller
from pynput.mouse import Controller
mouse = Controller()

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.7, min_tracking_confidence=0.7
)

# Add a custom CSS for background image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://static.vecteezy.com/system/resources/thumbnails/001/963/683/small/abstract-technology-background-background-3d-grid-cyber-technology-ai-tech-wire-network-futuristic-wireframe-artificial-intelligence-cyber-security-background-vector.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stToolbar"] {
    visibility: hidden;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title and Instructions
st.title("Virtual Mouse Using Hand Gesture")
st.write("Control your computer using hand gestures. Use the buttons below to Start or Stop the webcam.")

# Add inline Start and Stop buttons
col1, col2 = st.columns([1, 1])
start_webcam = col1.button("Start Webcam")
stop_webcam = col2.button("Stop Webcam")

# Video capture logic
if start_webcam:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Placeholder for video frames

    # Loop to process webcam feed
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = hands.process(frameRGB)

        if processed.multi_hand_landmarks:
            for hand_landmarks in processed.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                project.detect_gestures(frame, landmarks_list, processed)

        # Display the frame
        stframe.image(frame, channels="BGR")

        # Check for the stop button state
        if stop_webcam:
            st.write("Webcam Stopped.")
            break

    cap.release()
    stframe.empty()  # Clear the placeholder
    hands.close()
    st.write("Application has been stopped.")

