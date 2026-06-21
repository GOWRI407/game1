"""
gesture_recognizer.py
----------------------
Runs webcam capture + MediaPipe hand tracking in a background thread.
Classifies simple hand gestures and exposes the latest detected gesture
through a thread-safe shared object that the game loop can poll every frame.

Gestures recognized:
    OPEN_PALM   -> move forward
    FIST        -> stop
    POINT_LEFT  -> turn / move left
    POINT_RIGHT -> turn / move right
    THUMBS_UP   -> jump / interact
    NONE        -> no hand detected
"""

import threading
import time

import cv2
import mediapipe as mp


class GestureState:
    """Thread-safe container for the latest recognized gesture."""

    def __init__(self):
        self._lock = threading.Lock()
        self._gesture = "NONE"
        self._frame = None  # latest annotated frame (for optional preview window)

    def set(self, gesture, frame=None):
        with self._lock:
            self._gesture = gesture
            if frame is not None:
                self._frame = frame

    def get(self):
        with self._lock:
            return self._gesture

    def get_frame(self):
        with self._lock:
            return self._frame


def _finger_states(hand_landmarks, handedness_label):
    """
    Returns a list of booleans [thumb, index, middle, ring, pinky]
    indicating whether each finger is extended.
    """
    lm = hand_landmarks.landmark
    fingers = []

    # Thumb: compare x of tip vs joint (flip logic depending on hand)
    if handedness_label == "Right":
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)

    # Other 4 fingers: tip is above (smaller y) than pip joint -> extended
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(lm[tip].y < lm[pip].y)

    return fingers  # [thumb, index, middle, ring, pinky]


def classify_gesture(hand_landmarks, handedness_label):
    thumb, index, middle, ring, pinky = _finger_states(hand_landmarks, handedness_label)
    extended_count = sum([thumb, index, middle, ring, pinky])

    # Fist: nothing extended
    if extended_count <= 1 and not thumb:
        return "FIST"

    # Thumbs up: only thumb extended, hand oriented vertically
    if thumb and not index and not middle and not ring and not pinky:
        return "THUMBS_UP"

    # Open palm: all fingers extended
    if extended_count >= 4:
        return "OPEN_PALM"

    # Pointing: only index extended -> use wrist-to-index x direction for left/right
    if index and not middle and not ring and not pinky:
        wrist_x = hand_landmarks.landmark[0].x
        index_tip_x = hand_landmarks.landmark[8].x
        if index_tip_x < wrist_x - 0.05:
            return "POINT_LEFT"
        elif index_tip_x > wrist_x + 0.05:
            return "POINT_RIGHT"
        else:
            return "OPEN_PALM"

    return "NONE"


def run_gesture_thread(state: GestureState, show_preview=True, camera_index=0):
    """
    Background worker: captures webcam frames, runs MediaPipe hand detection,
    classifies the gesture, and updates `state`. Intended to be run via
    threading.Thread(target=run_gesture_thread, args=(state,), daemon=True).
    """
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[gesture_recognizer] ERROR: could not open webcam.")
        state.set("NONE")
        return

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture = "NONE"
            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness_label = results.multi_handedness[0].classification[0].label
                gesture = classify_gesture(hand_landmarks, handedness_label)

                if show_preview:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if show_preview:
                cv2.putText(
                    frame, f"Gesture: {gesture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )
                cv2.imshow("Gesture Control (press Q to quit camera view)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            state.set(gesture, frame)

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
