# UI libraries
from tkinter import *  # For creating GUI windows
import customtkinter as ctk  # For creating advanced UIs
from PIL import ImageTk, Image  # For dealing with images in GUI

# General libraries
import pickle  # For accessing the stored model with .p extension
import sys  # For closing the app when button clicked
from playsound import playsound  # Playing the corresponding sounds to the detected character

# ML libraries
import cv2  # For getting camera frames
from keras.preprocessing.sequence import pad_sequences  # For padding the detected data to remove whitespaces
import mediapipe as mp  # To detect hand landmarks
import numpy as np  # For dealing with arrays

# Audio recognition
import speech_recognition as sr  # For audio processing and recording audio

model_dict = pickle.load(open("model2.p", "rb"))  # Loading the model
model = model_dict["model"]

data = [("a", "sign_dataset/a.png"), ("b", "sign_dataset/b.png"), ("c", "sign_dataset/c.png"),
        ("d", "sign_dataset/d.png"), ("e", "sign_dataset/e.png"), ("f", "sign_dataset/f.png"),
        ("g", "sign_dataset/g.png"), ("h", "sign_dataset/h.png"), ("i", "sign_dataset/i.png"),
        ("j", "sign_dataset/j.png"), ("k", "sign_dataset/k.png"), ("l", "sign_dataset/l.png"),
        ("m", "sign_dataset/m.png"), ("n", "sign_dataset/n.png"), ("o", "sign_dataset/o.png"),
        ("p", "sign_dataset/p.png"), ("q", "sign_dataset/q.png"), ("r", "sign_dataset/r.png"),
        ("s", "sign_dataset/s.png"), ("t", "sign_dataset/t.png"), ("u", "sign_dataset/u.png"),
        ("v", "sign_dataset/v.png"), ("w", "sign_dataset/w.png"), ("x", "sign_dataset/x.png"),
        ("y", "sign_dataset/y.png"), ("z", "sign_dataset/z.png")]


def nlp():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        try:
            audio_data = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio_data)
            token = 2
        except sr.UnknownValueError:
            text = "Could not understand audio."
            token = 1
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            token = 0

    return token, text


def sign_to_speech():
    cap = cv2.VideoCapture(0)

    # Creating mediapipe objects
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=True,
        min_detection_confidence=0.5,
    )

    labels_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G",
                   7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N",
                   14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U",
                   21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
                   }

    i = 0

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        H, Width, _ = frame.shape

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

            x1 = int(min(x_) * Width) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * Width) - 10
            y2 = int(max(y_) * H) - 10

            # Getting the prediction from the trained model
            prediction = model.predict([np.asarray(data_aux)])
            character = labels_dict[int(prediction[0])]

            # Drawing rectangle and adding text
            cv2.rectangle(
                frame,
                (x1, y1), (x2, y2),
                (0, 0, 0), 4
            )
            cv2.putText(
                frame, character,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 0), 3, cv2.LINE_AA
            )

            # Playing the corresponding sound
            try:
                if character != previous_character or i == 0:
                    playsound(f"audio_files/{character}.mp3")
            except NameError:
                pass

            cv2.imshow("Sign to Speech Converter", frame)

            previous_character = character
            i = 1

        # 'q' key to stop the loop
        if cv2.waitKey(1000) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def start_speech_procedure():
    button2.configure(text="Listening...", fg_color="green")
    button2.after(100, speech_to_sign)


def speech_to_sign():
    win2 = Toplevel()
    win2.geometry("900x300")
    win2.title("Sign Sense - Speech2Sign")

    token, text = nlp()

    if token == 2:
        text = text.split(" ")[0]
        x = 50

        for letter in text:
            for element in data:
                if element[0] == letter.lower():
                    image_path = element[1]

            image1 = Image.open(image_path)
            original_width, original_height = image1.width, image1.height

            resized_image = image1.resize((original_width // 2, original_height // 2), Image.LANCZOS)
            test = ImageTk.PhotoImage(resized_image)
            label1 = Label(master=win2, image=test)
            label1.image = test

            if letter.lower() == "a" or letter.lower() == "e" or letter.lower() == "s":
                label1.place(x=x, y="30")
            else:
                label1.place(x=x, y="45")

            label_text = Label(master=win2, text=letter.upper())
            label_text.place(x=x + (original_width // 4), y="10")

            x += (original_width // 2 + 50)

    label2 = ctk.CTkLabel(master=win2,
                          text=text,
                          width=120,
                          height=25,
                          corner_radius=8,
                          font=("Montserrat", 14),
                          text_color="darkgray")
    label2.place(relx=0.5, rely=0.8, anchor=CENTER)

    button2.configure(fg_color=orig_color, text="Speech2Sign")

    win2.mainloop()


# Creating GUI
win = Tk()
win.geometry("550x600")
win.title("Sign Sense")

close_img = ctk.CTkImage(
    Image.open("images/close.png"),
    size=(20, 20)
)
logo_img = ImageTk.PhotoImage(Image.open("images/logo.png"))
panel = Label(win, image=logo_img)
panel.pack(side="top", fill="both")

label = ctk.CTkLabel(master=win,
                     text="You may need to adjust your hand a little bit to get a perfect output...",
                     width=120,
                     height=25,
                     corner_radius=8,
                     font=("Montserrat", 11),
                     text_color="darkgray")
label.place(relx=0.5, rely=0.87, anchor=CENTER)

button = ctk.CTkButton(master=win,
                       text="Sign2Speech",
                       command=sign_to_speech,
                       width=120,
                       height=40,
                       border_width=0,
                       corner_radius=8,
                       font=("Montserrat", 14))
button.place(relx=0.3, rely=0.94, anchor=CENTER)

button2 = ctk.CTkButton(master=win,
                        text="Speech2Sign",
                        command=start_speech_procedure,
                        width=120,
                        height=40,
                        border_width=0,
                        corner_radius=8,
                        font=("Montserrat", 14))
button2.place(relx=0.7, rely=0.94, anchor=CENTER)
orig_color = button2.cget("fg_color")

close_button = ctk.CTkButton(master=win,
                             text="",
                             command=sys.exit,
                             image=close_img,
                             fg_color="white",
                             hover_color="white",
                             width=10,
                             height=10,
                             corner_radius=5)
close_button.place(relx=0.9, rely=0.05, anchor=CENTER)

win.mainloop()
