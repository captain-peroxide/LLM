import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from os.path import isfile

# Set the duration and sample rate of the recording
duration = 30  # seconds
sample_rate = 16000  # Hz

recording_file = "recording1.wav"

if isfile(recording_file):
    print(f"{recording_file} already exists. If you want to create another recording with the same name, delete it first.")
else:
    # Record audio
    print("Starting recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until the recording is finished
    print("Recording finished.")

    # Convert the audio to 16-bit data
    audio = np.int16(audio * 32767)

    # Write the audio data to a .wav file
    write(recording_file, sample_rate, audio)