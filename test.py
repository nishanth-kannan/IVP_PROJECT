# importing dependencies required for the project
import cv2
import mediapipe as mp
import time
import handTracking2 as ht2

# To calculate fps of the video stream
prev_time, current_time = 0, 0

# Opening Camera
cap = cv2.VideoCapture(0)

hand_detector = ht2.hand_detection()

while True:
    success, frame = cap.read()

    frame = hand_detector.find_hands(frame, to_draw=True)
    landmarks_list = hand_detector.find_landmarks(frame, to_draw=True)

    # Calculating the frame rate of the video stream
    current_time = time.time()
    fps = int(1 / (current_time - prev_time))
    prev_time = current_time

    # Flipping the image so there is no confusion of the orientation of camera
    frame = cv2.flip(frame, 1)

    # Text on video stream
    cv2.putText(frame, str(fps), (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (47, 255, 173), 3)

    # Video stream
    cv2.imshow("Image", frame)
    cv2.waitKey(1)