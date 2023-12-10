import os
import pickle
import mediapipe as mp
import cv2

# Creating mediapipe objects
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)

data = []
labels = []

for dir_ in os.listdir("./data"):       # for each of the directory in data folder:
    if dir_ != ".DS_Store":     # ignoring the .DS_Store
        print(f"Processing directory: {dir_}")
        for img_path in os.listdir(os.path.join("./data", dir_)):       # for each of the images in the sub-folders
            print(f"Processing image: {img_path}")
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join("./data", dir_, img_path))        # Reading the image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # Converting it to RGB format from BGR

            results = hands.process(img_rgb)        # Getting the landmarks
            if results.multi_hand_landmarks:        # if landmarks detected:
                print(f"Hand landmarks detected in {img_path}")
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

                data.append(data_aux)   # Appending the x, y co-ordinates to data
                labels.append(dir_)     # Appending the respective labels

file = open('data.pickle', 'wb')        # Saving the data in binary format in data.pickle
pickle.dump({'data': data, 'labels': labels}, file)
file.close()
