from tkinter import *  # For creating GUI windows
import customtkinter as ctk  # For creating advanced UIs
from PIL import ImageTk, Image  # For dealing with images in GUI
from pygame import mixer  # Playing the corresponding sounds to the detected character
import cv2  # For getting camera frames
from audio_processing import nlp  # Library to transcribe audio
from process_img import process  # Local library to detect landmarks on hands
import os  # Useful for working with local directories
import automated_video_gen as avg  # To generate videos by concatenation frames

# Loading SSL certificate
os.environ['SSL_CERT_FILE'] = ("/Users/ajaygautam/Desktop/Sign-Sense-main/venv/lib/python3.12/site-packages/certifi"
                               "/cacert.pem")


def sign_to_speech():
    """
    Processes frames and detects hand in them using Mediapipe library. Then performs series of pre-defined
    algorithms to detect ASL sign using hand landmarks.

    :return: Continuous procedure to detect hand and predict sign till 'Q' is not pressed
    """
    cap = cv2.VideoCapture(0)

    i = 0
    while True:
        ret, frame = cap.read()
        height, width, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detecting hands in frame and predicting sign using pre-trained AI model
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
                    """
                        Multiple checks:
                            1. Checks whether the prediction is being continuously repeated to avoid 'bad' noise.
                            2. Checks if it is the first letter, if it is, then play the sound manually to avoid being
                               missed.
                    """
                    mixer.init()

                    try:
                        mixer.music.load(f"audio_files/{character}.mp3")
                        mixer.music.play()
                    except FileNotFoundError:
                        print(f"Audio file for character '{character}' not found.")
                    except Exception as e:
                        print(f"Error playing sound: {e}")

                    mixer.quit()

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
    """
    Transcribes recorded audio using OpenAI Whisper API and then uses 'PIL' and 'ffmpeg' to concatenate images and
    create a video of corresponding ASL signs.

    :return: Continuous procedure to transcribe audio and show AI generated video of ASL signs
    """

    # Editing the generated video to fit screen, dynamically
    class VideoPlayer(Frame):
        def __init__(self, parent, video_path, width, height):
            super().__init__(parent)
            self.video_path = video_path
            self.width = width
            self.height = height
            self.cap = cv2.VideoCapture(video_path)
            self.lbl = Label(self)
            self.lbl.pack()
            self.update_video()

        def update_video(self):
            ret, frame = self.cap.read()
            if ret:
                # Resize the frame
                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.lbl.imgtk = imgtk  # Keep a reference to avoid garbage collection
                self.lbl.configure(image=imgtk)
            self.after(33, self.update_video)

    win2 = Toplevel()
    win2.geometry("800x400")
    win2.title("Sign Sense - Speech2Sign")

    sen_type, text = nlp()      # Transcribe recorded audio using OpenAI API
    text = text.split(" ")
    letters = "abcdefghijklmnopqrstuvwxyz"

    image_filenames = []
    for word in text:
        for letter in word:
            if letter.lower() in letters:
                image_filenames.append(f"sign_dataset/{letter.lower()}.png")

    # Concatenating images to create video
    avg.create_video_with_ffmpeg(image_filenames, "final_video.mp4", duration_per_image=2)

    # Displaying images on screen
    video_player = VideoPlayer(win2, "final_video.mp4", width=500, height=300)
    video_player.pack()

    if sen_type == "Interrogative":         # Sentence type-check
        text += "?"

    label2 = ctk.CTkLabel(master=win2, text=text,
                          width=120, height=25, corner_radius=8,
                          font=("Montserrat", 14), text_color="darkgray")
    label2.pack()

    button2.configure(fg_color=orig_color, text="Speech2Sign")

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
