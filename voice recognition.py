import cv2
import numpy as np
import pandas as pd
import speech_recognition as sr
import pyaudio
import wave
import threading
import tkinter as tk
from tkinter import ttk
from pydub import AudioSegment
from pydub.effects import normalize
from PIL import Image, ImageTk
import noisereduce as nr
import soundfile as sf
from scipy.signal import butter, lfilter

# Initialize the recognizer
recognizer = sr.Recognizer()

# Global flag to control thread execution
running = True

# Butterworth bandpass filter for noise reduction
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Function to capture audio from the microphone and convert it to text
def capture_audio_and_recognize():
    # Set the parameters for the audio capture
    chunk_size = 1024  # Experiment with different chunk sizes
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = 10  # Increased duration of recording

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    try:
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk_size,
                        input=True)

        while running:
            frames = []  # Initialize array to store frames

            # Store data in chunks for the given duration
            for _ in range(0, int(fs / chunk_size * seconds)):
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)

            # Convert the raw data to an audio segment
            raw_audio = AudioSegment(
                data=b''.join(frames),
                sample_width=p.get_sample_size(sample_format),
                frame_rate=fs,
                channels=channels
            )

            # Preprocess the audio to improve recognition accuracy
            # Normalize the audio
            normalized_audio = normalize(raw_audio)

            # Increase volume significantly
            gain_dB = 30  # Adjust this value as needed
            louder_audio = normalized_audio.apply_gain(gain_dB)

            # Convert audio segment to numpy array for noise reduction
            audio_data = np.array(louder_audio.get_array_of_samples())
            
            # Apply bandpass filter
            filtered_audio_data = bandpass_filter(audio_data, lowcut=300.0, highcut=3400.0, fs=fs, order=6)

            # Apply noise reduction
            reduced_noise_audio_data = nr.reduce_noise(y=filtered_audio_data, sr=fs)

            # Convert numpy array back to audio segment
            reduced_noise_audio = AudioSegment(
                reduced_noise_audio_data.tobytes(),
                frame_rate=fs,
                sample_width=louder_audio.sample_width,
                channels=louder_audio.channels
            )

            # Save the processed audio as a WAV file
            wav_file = "output.wav"
            reduced_noise_audio.export(wav_file, format="wav")

            # Recognize speech using the saved WAV file
            with sr.AudioFile(wav_file) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    print(f"Recognized text: {text}")  # Debugging: Print recognized text
                    if text:
                        text_area.insert(tk.END, text + "\n")
                        text_area.see(tk.END)
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")  # Debugging
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")  # Debugging

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

    except OSError as e:
        print(f"Error capturing audio: {e}")

def update_camera_feed():
    cap = cv2.VideoCapture(0)
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image from OpenCV format to ImageTk format
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the image in the label
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

    cap.release()

def on_closing():
    global running
    running = False
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Camera and Speech Recognition")
root.protocol("WM_DELETE_WINDOW", on_closing)

# Create a text area to display recognized words
text_area = tk.Text(root, wrap='word', width=50)
text_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Create a label to display the camera feed
camera_label = ttk.Label(root)
camera_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Start the threads for updating the text area and camera feed
threading.Thread(target=capture_audio_and_recognize, daemon=True).start()
threading.Thread(target=update_camera_feed, daemon=True).start()

# Start the Tkinter main loop
root.mainloop()




