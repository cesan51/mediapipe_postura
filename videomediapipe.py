import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture("a1.mp4")

with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(frame_rgb)

        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 225, 225), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 128, 225), thickness=2))

        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 225, 225), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 128, 225), thickness=2))

        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 225, 225), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 128, 225), thickness=2))

        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 225, 225), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 128, 225), thickness=2))

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

       # mp_drawing.plot_landmarks(
        #    frame, results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

cv2.destroyAllWindows()
