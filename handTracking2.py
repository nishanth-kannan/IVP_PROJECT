# importing dependencies required for the project
import cv2
import mediapipe as mp
import time

class hand_detection():
    def __init__(self):
        # Loading the Mediapipe Hands Module and storing them in variables
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

def find_hands(self, frame, to_draw = False):
    # hands.process method detects the hand in the frame and assigns all the landmarks on the hand with x, y, z coordinates
    self.results = self.hands.process(frame)

    # Coordinates of the landmarks
    # print(results.multi_hand_landmarks)

    if self.results.multi_hand_landmarks:
        for landmarks in self.results.multi_hand_landmarks:
            if to_draw:
                self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
    
    return frame

def find_landmarks(self, frame, hand = 0, to_draw = False):
    landmarks_list = []

    if self.results.multi_hand_landmarks:
        hand_in_frame = self.results.multi_hand_landmarks[hand]
        for id, lm in enumerate(hand_in_frame.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)

            print("Hand Detected in frame")
            landmarks_list.append([id, cx, cy])

            if to_draw:
                cv2.circle(frame, (cx, cy), 12, (47, 255, 173), cv2.FILLED)

    return landmarks_list


def main():
    # To calculate fps of the video stream
    prev_time, current_time = 0, 0

    # Opening Camera
    cap = cv2.VideoCapture(0)

    hand_detector = hand_detection()

    while True:
        success, frame = cap.read()

        if not success:
            print("Cannot access Camera")
            break
        
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


if __name__ == "__main__":
    main()

