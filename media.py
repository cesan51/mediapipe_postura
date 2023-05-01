import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(
    static_image_mode = True,
    model_complexity=2) as holistic:

    image = cv2.imread("image1.jpeg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = holistic.process(image_rgb)

    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 225, 225), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 128, 225), thickness=2
    ))

    # Mano Izquierda

    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 225, 225), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 128, 225), thickness=2
    ))

    # Mano Derecha

    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 225, 225), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 128, 225), thickness=2
    ))

    # Cuerpo

    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 225, 225), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 128, 225), thickness=2
    ))

   



    cv2.imshow("Image", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()