# UI libraries
import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk, Image

# General libraries
import cv2
import pickle
from playsound import playsound

# ML libraries
from keras.preprocessing.sequence import pad_sequences
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open("model2.p", "rb"))  # Loading the model
model = model_dict["model"]

# Creating GUI
win = tk.Tk()
win.geometry("550x600")
win.title("Sign Sense")
ctk.set_appearance_mode("Dark")

img = ImageTk.PhotoImage(Image.open("images/logo.png"))
panel = tk.Label(win, image=img)
panel.pack(side="top", fill="both")

label = ctk.CTkLabel(master=win,
                     text="You may need to adjust your hand a little bit to get a perfect output...",
                     width=120,
                     height=25,
                     corner_radius=8,
                     font=("Montserrat", 11),
                     text_color="darkgray")
label.place(relx=0.5, rely=0.87, anchor=tk.CENTER)

button = ctk.CTkButton(master=win,
                       text="Let's Begin",
                       command=win.destroy,
                       width=120,
                       height=40,
                       border_width=0,
                       corner_radius=8,
                       font=("Montserrat", 14))
button.place(relx=0.5, rely=0.94, anchor=tk.CENTER)

win.mainloop()

cap = cv2.VideoCapture(0)

# Creating mediapipe objects
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)

labels_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G",
               7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N",
               14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U",
               21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
               }

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Getting the landmarks
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

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Getting the prediction from the trained model
        prediction = model.predict([np.asarray(data_aux)])
        character = labels_dict[int(prediction[0])]

        # Drawing rectangle and adding text
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),
            4
        )
        cv2.putText(
            frame,
            character,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )

        # Playing the corresponding sound
        try:
            if character != previous_character:
                playsound(f"audio_files/{character}.mp3")
        except NameError:
            pass

        cv2.imshow("Sign to Speech Converter", frame)

        previous_character = character

    # 'q' key to stop the loop
    if cv2.waitKey(100) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
