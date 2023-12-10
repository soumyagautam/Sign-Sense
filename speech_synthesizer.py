from gtts import gTTS       # gTTS: Google Text-To-Speech
import os

alphabets = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z"
]

os.makedirs(os.path.join("kivyApp/audio_files"))        # Creating audio_files directory

for word in alphabets:
    tts = gTTS(word)       # Getting the speech for each alphabet
    tts.save(f"audio_files/{word}.mp3")     # Saving each audio file in .mp3 format
