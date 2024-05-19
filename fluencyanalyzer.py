import librosa
import numpy as np
import noisereduce as nr

from faster_whisper import WhisperModel

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

class FluencyAnalyzer:
    def _init_(self):
        pass

    def calculate_rms_energy(self, y, frame_length=2048, hop_length=512):
        try:
            num_frames = len(y) // hop_length
            rms_energy_array = np.zeros(num_frames)

            for i in range(num_frames):
                start = i * hop_length
                end = start + frame_length
                frame = y[start:end]
                rms_energy_array[i] = np.sqrt(np.mean(frame ** 2))

            rms_energy_norm = (rms_energy_array - np.min(rms_energy_array)) / (np.max(rms_energy_array) - np.min(rms_energy_array) + 0.01)

            return rms_energy_norm
        except Exception as e:
            print("Error in calculating RMS energy:", e)
            return None

    def calculate_pitch_variance(self, y, sr, frame_length=2048, hop_length=512, threshold=1000):
        try:
            pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)

            pitch_variation = np.sqrt(np.var(pitches[pitches > threshold]))

            return pitch_variation / (np.max(pitches)+0.1)
        except Exception as e:
            print("Error in calculating pitch variance:", e)
            return None

    def calculate_wpm(self, segments):
        try:
            seg = []
            start = []
            end = []
            for segment in segments:
                seg.append(segment.text)
                start.append(segment.start)
                end.append(segment.end)

            leng = [len(sent.split()) for sent in seg]

            wpm = [(l*60)/(e-s) for l,s,e in zip(leng,start,end)]

            return np.average(wpm)
        except Exception as e:
            print("Error in calculating words per minute (WPM):", e)
            return None

    def calculate_fluency_confidence(self, audio_file, segments):
        try:
            y, sr = librosa.load(audio_file, sr=None)
            y = nr.reduce_noise(y=y, sr=sr)
            rms_energy_norm = self.calculate_rms_energy(y)
            if rms_energy_norm is None:
                return None, None

            rms_energy = np.mean(rms_energy_norm)

            wpm = self.calculate_wpm(segments)
            if wpm is None:
                return None, None

            pitch_variance = self.calculate_pitch_variance(y, sr)
            if pitch_variance is None:
                return None, None

            # pitch_variance_norm = 1 - pitch_variance
            pitch_variance_norm = np.exp(-0.8*np.sqrt(pitch_variance))

            fluency_score = (0.2*rms_energy + 0.5*(wpm / 120) + 0.3*(pitch_variance_norm))
            confidence_score = (0.1*rms_energy + 0.9*pitch_variance_norm)

            return fluency_score, confidence_score
        except Exception as e:
            print("Error in calculating fluency and confidence:", e)
            return None, None
        

audio_file = "C:\Users\user\OneDrive\Desktop\llm\WhatsApp Audio 2024-04-12 at 18.01.36_510e2a73.mp3"
segments, info = model.transcribe(audio_file, beam_size=5)
Scorer = FluencyAnalyzer()
fluency, confidence = Scorer.calculate_fluency_confidence(audio_file,segments)
print("Fluency Score:", min(10,10*fluency))
print("Confidence Score:", min(10,10*(confidence)))        