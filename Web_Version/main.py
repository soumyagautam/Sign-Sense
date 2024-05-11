from flask import Flask, render_template
import cv2
from process_img import process
from audio_processing_web import nlp

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/si2sp", methods=["POST"])
def si2sp():
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    height, width, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    code, character, hand_pos = process(frame, frame_rgb, width, height)

    if code:
        cv2.rectangle(
            frame,
            (hand_pos[0], hand_pos[1]), (hand_pos[2], hand_pos[3]),
            (0, 0, 0), 4
        )
        cv2.putText(
            frame, character, (hand_pos[0], hand_pos[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA
        )

    cv2.imwrite("static/img.png", frame)

    return render_template("si2sp.html", img="./static/img.png")


@app.route("/sp2te", methods=["POST"])
def sp2te():
    text = nlp()

    return render_template("index.html", text=text)
    

@app.route("/info.html")
def info():
    return render_template("info.html")


if __name__ == "__main__":
    app.run(debug=True)
