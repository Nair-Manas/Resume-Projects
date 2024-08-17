

import cv2
import mediapipe as mp
import pyautogui
import math

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize Mediapipe Drawing Utils
drawing_utils = mp.solutions.drawing_utils

# Get the screen width and height
screen_width, screen_height = pyautogui.size()

# Variables for smoothening the mouse movement
smoothening = 9
plocx, plocy = 0, 0
clocx, clocy = 0, 0

# Click threshold distance between index finger tip and thumb tip
click_threshold = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get hand landmarks
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_x, index_y = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)

            # Get the thumb tip coordinates
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_x, thumb_y = int(thumb_tip.x * screen_width), int(thumb_tip.y * screen_height)

            # Smooth the mouse movement
            clocx = plocx + (index_x - plocx) / smoothening
            clocy = plocy + (index_y - plocy) / smoothening

            # Move the mouse cursor
            pyautogui.moveTo(clocx, clocy)

            # Check if the distance between index finger tip and thumb tip is below the click threshold
            distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)
            if distance < click_threshold:
                pyautogui.click()

            # Update the previous location
            plocx, plocy = clocx, clocy

    # Show the frame with landmarks
    cv2.imshow('Virtual Mouse', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

