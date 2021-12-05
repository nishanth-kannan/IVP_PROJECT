# importing dependencies required for the project
import cv2
import mediapipe as mp
import time

# Oepning Camera
cap = cv2.VideoCapture(0)
 
# Loading the Mediapipe Hands Module and storing them in variables
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
 
# To calculate fps of the video stream
prev_time, current_time = 0, 0
hand_not_in_frame_count = 0

while True:
    # Reading each frame 
    success, frame = cap.read()

    # If opening Camera is not possible
    if not success:
      print("Cannot access Camera")
      break

    # hands.process method detects the hand in the frame and assigns all the landmarks on the hand with x, y, z coordinates
    results = hands.process(frame)

    # Coordinates of the landmarks
    # print(results.multi_hand_landmarks) 

    # Puts circles on the hand, marks all the landmarks on the hand
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(landmarks.landmark):
                # print(id, lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Displaying coordinates of all 21 landmarks (id (0 - 20), x-coordinate, y-coordinate)
                # print(id, cx, cy)
                print("Hand Detected in frame")
                cv2.circle(frame, (cx, cy), 12, (47, 255, 173), cv2.FILLED)
                hand_not_in_frame_count = 0 # resets count
 
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
 
    # if hand is not in frame for a long period of time, program exits
    else:
        hand_not_in_frame_count += 1
        # count increases for each frame that hand is not detected, so depending on your frame rate, cooldown might vary and program might exit
        # cooldown time = 500 / fps (in seconds)
        if(hand_not_in_frame_count > 500):
            break

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