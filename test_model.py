# UI libraries
from tkinter import *  # For creating GUI windows
import customtkinter as ctk  # For creating advanced UIs
from PIL import ImageTk, Image  # For dealing with images in GUI
from pygame import mixer  # Playing the corresponding sounds to the detected character

# ML libraries
import cv2  # For getting camera frames
from audio_processing import nlp  # Library to transcribe audio
from process_img import process


def sign_to_speech():
    cap = cv2.VideoCapture(0)

    i = 0
    while True:
        ret, frame = cap.read()
        height, width, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        code, character, hand_pos = process(frame, frame_rgb, width, height)

        if code:
            # Drawing rectangle and adding text
            cv2.rectangle(
                frame,
                (hand_pos[0], hand_pos[1]), (hand_pos[2], hand_pos[3]),
                (0, 0, 0), 4
            )
            cv2.putText(
                frame, character, (hand_pos[0], hand_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA
            )

            # Playing the corresponding sound
            try:
                if character != previous_character or i == 0:
                    mixer.init()
                    mixer.music.load(f"audio_files/{character}.mp3")
                    mixer.music.play()
            except NameError:
                pass

            previous_character = character
            i = 1

        cv2.imshow("Sign Sense - Sign2Speech", frame)

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

    sen_type, text = nlp()            # Natural Language Processing (NLP)

    text = text.split(" ")
    x = 50
    y = 75

    main_frame = Frame(win2)
    main_frame.pack(fill=BOTH, expand=1)

    my_canvas = Canvas(main_frame)
    my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

    my_scrollbar = Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
    my_scrollbar.pack(side=RIGHT, fill=Y)

    my_canvas.configure(yscrollcommand=my_scrollbar.set)
    my_canvas.bind(
        '<Configure>', lambda e: my_canvas.configure(scrollregion=my_canvas.bbox("all"))
    )

    def _on_mousewheel(event):
        my_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    my_canvas.bind_all("<MouseWheel>", _on_mousewheel)

    second_frame = Frame(my_canvas, width=1000, height=1000)
    second_frame.pack(fill=BOTH, expand=True)

    letters = []
    for letter in range(ord("A"), ord("z") + 1):
        letters.append(chr(letter))

    for word in text:
        for letter in word:
            if letter in letters:
                image_path = f"sign_dataset/{letter}.png"

            try:
                image1 = Image.open(image_path)
                original_width, original_height = image1.width, image1.height

                resized_image = image1.resize((original_width // 3, original_height // 3), Image.LANCZOS)
                test = ImageTk.PhotoImage(resized_image)
                label1 = Label(master=second_frame, image=test)
                label1.image = test

                if letter.lower() == "a" or letter.lower() == "e" or letter.lower() == "s" or letter.lower() == "r":
                    label1.place(x=x - 20, y=y - 10)
                else:
                    label1.place(x=x, y=y)

                label_text = Label(master=second_frame, text=letter.upper())
                label_text.place(x=x + (original_width // 6), y=y - 25)

                x += (original_width // 2 + 50)
            except UnboundLocalError:
                text = "Error"

        y += 150
        x = 50

    if sen_type == "Interrogative":
        text += "?"

    label2 = ctk.CTkLabel(master=second_frame, text=text,
                          width=120, height=25, corner_radius=8,
                          font=("Montserrat", 14), text_color="darkgray")
    label2.place(x=80, y=30, anchor="w")

    button2.configure(fg_color=orig_color, text="Speech2Sign")

    my_canvas.create_window((0, 0), window=second_frame, anchor="nw")
    second_frame.configure(height=y)

    win2.mainloop()


# Creating GUI
win = Tk()
win.geometry("550x600")
win.title("Sign Sense")

logo_img = ImageTk.PhotoImage(Image.open("images/logo.png"))
panel = Label(win, image=logo_img)
panel.pack(side="top", fill="both")

label = ctk.CTkLabel(master=win, text="Sign Sense : v1.3.2",
                     width=120, height=25, corner_radius=8,
                     font=("Montserrat", 11), text_color="darkgray")
label.place(relx=0.5, rely=0.87, anchor=CENTER)

button = ctk.CTkButton(master=win, text="Sign2Speech",
                       command=sign_to_speech,
                       width=120, height=40,
                       border_width=0, corner_radius=8,
                       font=("Montserrat", 14))
button.place(relx=0.3, rely=0.94, anchor=CENTER)

button2 = ctk.CTkButton(master=win, text="Speech2Sign",
                        command=start_speech_procedure,
                        width=120, height=40,
                        border_width=0, corner_radius=8,
                        font=("Montserrat", 14))
button2.place(relx=0.7, rely=0.94, anchor=CENTER)
orig_color = button2.cget("fg_color")


def on_hover1(event):
    label.configure(text="Try adjusting your hand a little bit to get a perfect output...")


def on_leave1(event):
    label.configure(text="Sign Sense : v1.3.2")


def on_hover2(event):
    label.configure(text="Start speaking 1 second after the listening process starts for better results...")


def on_leave2(event):
    label.configure(text="Sign Sense : v1.3.2")


button.bind("<Enter>", on_hover1)
button.bind("<Leave>", on_leave1)

button2.bind("<Enter>", on_hover2)
button2.bind("<Leave>", on_leave2)

win.mainloop()
