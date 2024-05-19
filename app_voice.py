from flask import Flask, request, jsonify
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import json
app = Flask(__name__)

# Load the Whisper model
processor = Wav2Vec2Processor.from_pretrained("openai/whisper-large-v3")
model = Wav2Vec2ForCTC.from_pretrained("openai/whisper-large-v3")

@app.route("/speech-recognition", methods=["POST"])
def speech_recognition():
    # Get the audio data from the request
    audio_data = request.files["audio"].read()
    print(audio_data)

    # Perform speech recognition
    input_values = processor(audio_data, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    with open("./transcription.json", "w") as f:
        json.dump({"transcription": transcription}, f)


    return jsonify({"transcription": transcription})

if __name__ == "_main_":
    app.run(host="0.0.0.0", port=5000)