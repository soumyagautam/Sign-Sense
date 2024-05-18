import speech_recognition as sr  # For audio processing and recording audio
import wave  # To create a .wav file out of the microphone recordings
import whisper  # Advanced OpenAI audio transcribing model
import librosa  # Converting the .wav file into a floating-point time series
import numpy as np


def nlp():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            audio_data = recognizer.listen(source, timeout=10)

        with wave.open("./static/captured_audio1.wav", "wb") as wave_file:
            wave_file.setnchannels(1)
            wave_file.setsampwidth(audio_data.sample_width)
            wave_file.setframerate(audio_data.sample_rate)
            wave_file.writeframes(audio_data.frame_data)

        y, source = librosa.load("./static/captured_audio1.wav")
        audio_waveform = np.array(y)

        audio_model = whisper.load_model("base")
        result = audio_model.transcribe(audio_waveform, fp16=False, language="en")["text"][1:]

        return result

    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from OpenAI Whisper API service; {e}")
        return None
