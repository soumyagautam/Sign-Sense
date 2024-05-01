import mediapipe as mp
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences  # For padding the detected data to remove whitespaces

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

model_dict = pickle.load(open("model3.p", "rb"))  # Loading the model
model = model_dict["model"]

labels_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G",
               7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N",
               14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U",
               21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"}


def process(frame, frame_rgb, width, height):
    data_aux = []
    x_ = []
    y_ = []

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Drawing the landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Padding the landmarks
        data_aux = pad_sequences([data_aux], maxlen=42, padding='post', truncating='post', dtype='float32')[0]

        x1 = int(min(x_) * width) - 10
        y1 = int(min(y_) * height) - 10

        x2 = int(max(x_) * width) - 10
        y2 = int(max(y_) * height) - 10

        # Getting the prediction from the trained model
        prediction = model.predict([np.asarray(data_aux)])
        character = labels_dict[int(prediction[0])]

        return True, character, (x1, y1, x2, y2)

    return None, None, None
