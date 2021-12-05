# importing dependencies required for the project
import cv2
import mediapipe as mp
import time
import numpy as np

class hand_detection():
    def __init__(self):
        # Loading the Mediapipe Hands Module and storing them in variables
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, to_draw=False):
        # hands.process method detects the hand in the frame and assigns all the landmarks on the hand with x, y, z coordinates
        self.results = self.hands.process(frame)

        # Coordinates of the landmarks
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                if to_draw:
                    self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

        return frame

    def find_landmarks(self, frame, hand=0, to_draw=False):
        landmarks_list = []

        if self.results.multi_hand_landmarks:
            hand_in_frame = self.results.multi_hand_landmarks[hand]
            for id, lm in enumerate(hand_in_frame.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # print("Hand Detected in frame")
                landmarks_list.append([id, cx, cy])

                if to_draw:
                    cv2.circle(frame, (cx, cy), 12, (47, 255, 173), cv2.FILLED)

        return landmarks_list

    def fingers_up(self, landmarks_list):
        fingers = []
        finger_tips = [4, 8, 12, 16, 20]  # finger tip landmarks from MediaPipe: https://google.github.io/mediapipe/solutions/hands.html

        if landmarks_list[finger_tips[0]][1] < landmarks_list[finger_tips[0] - 1][
            1]:  # Check if tip of thumb is right or left (horizontal)(considering right hand)
            fingers.append(1)
        else:
            fingers.append(0)

        for i in range(1, 5):
            if landmarks_list[finger_tips[i]][2] < landmarks_list[finger_tips[i] - 2][
                2]:  # Check if finger tip is up or down (vertical)
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():
    # Specify header menu to select color, eraser options
    imPath = "/home/nishanth/IVP Assignments/Project/Images/UI.png"
    header = cv2.imread(imPath)
    drawColor = (0, 0, 255)

    # To calculate fps of the video stream
    prev_time, current_time = 0, 0

    # Opening Camera
    cap = cv2.VideoCapture(0)

    hand_detector = hand_detection()

    x_prev, y_prev = 0, 0

    brush_thickness = 25
    eraser_thickness = 50

    drawing_canvas = np.zeros(shape=(480, 640, 3), dtype=np.uint8)

    while True:
        success, frame = cap.read()

        # Flipping the image so there is no confusion of the orientation of camera
        frame = cv2.flip(frame, 1)

        if not success:
            print("Cannot access Camera")
            break

        frame = hand_detector.find_hands(frame, to_draw=True)
        landmarks_list = hand_detector.find_landmarks(frame, to_draw=True)

        if len(landmarks_list) != 0:
            # tips of index and middle fingers
            x1, y1 = landmarks_list[8][1:]
            x2, y2 = landmarks_list[12][1:]

            fingers = hand_detector.fingers_up(landmarks_list)
            # print(fingers)

            # Both fingers are up, therefore selection mode
            if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                x_prev, y_prev = 0, 0
                print("Selection")
                # Checking for the click
                if y1 < 60:
                    if 0 < x1 < 160:
                        drawColor = (0, 0, 255)
                    elif 160 < x1 < 320:
                        drawColor = (255, 0, 0)
                    elif 320 < x1 < 480:
                        drawColor = (0, 255, 0)  #
                    elif 480 < x1 < 640:
                        drawColor = (0, 0, 0)  # Eraser
                cv2.rectangle(frame, (x1, y1 - 20), (x2, y2 + 20), drawColor, cv2.FILLED)  # rectangle for selection

            # Spiderman Hand - Clears everything on screen
            if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
                print("Clear All")
                cv2.rectangle(frame, (0, 0), (640, 480), (0, 0, 0), cv2.FILLED)
                cv2.rectangle(drawing_canvas, (0, 0), (640, 480), (0, 0, 0), cv2.FILLED)

            # Index finger is up, therefore drawing mode
            if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                print("Drawing")
                cv2.circle(frame, (x1, y1), 15, drawColor, cv2.FILLED)
                if x_prev == 0 and y_prev == 0:  # first frame of drawing (point is drawn at the same position)
                    x_prev, y_prev = x1, y1

                if drawColor == (0, 0, 0):  # eraser is selected (fill with background color)
                    cv2.line(frame, (x_prev, y_prev), (x1, y1), drawColor, eraser_thickness)  # draw line from x_prev, y_prev to current x1, y1
                    cv2.line(drawing_canvas, (x_prev, y_prev), (x1, y1), drawColor, eraser_thickness)
                else:
                    cv2.line(frame, (x_prev, y_prev), (x1, y1), drawColor, brush_thickness)  # draw line from x_prev, y_prev to current x1, y1
                    cv2.line(drawing_canvas, (x_prev, y_prev), (x1, y1), drawColor, brush_thickness)

                x_prev, y_prev = x1, y1

            #If not in drawing mode, resets x_prev and y_prev
            else:
                x_prev, y_prev = 0, 0

        # Calculating the frame rate of the video stream
        current_time = time.time()
        fps = int(1 / (current_time - prev_time))
        prev_time = current_time

        # Text on video stream
        cv2.putText(frame, str(fps), (10, 120), cv2.FONT_HERSHEY_PLAIN, 3, (47, 255, 173), 3)

        # Show drawing on main camera as well
        frame = cv2.addWeighted(frame, 1, drawing_canvas, 0.8, 0)
        frame[:55, :, :] = cv2.addWeighted(frame[:55, :, :], 0.2, header[:55, :, :], 1, 0)

        # Video stream
        cv2.imshow("Image", frame)
        cv2.imshow("Canvas", drawing_canvas)
        cv2.waitKey(1)

        # print(frame.shape)
        # print(drawing_canvas.shape)


if __name__ == "__main__":
    main()