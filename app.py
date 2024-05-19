import panel as pn
import warnings
warnings.filterwarnings("ignore")
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
import base64
import io
import wave
import csv


pn.extension()

MODEL_KWARGS = {
    "mistral": {
        "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "model_file": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    },
}

llm_chains = {}

TEMPLATE = """<s>[INST] As a technical job interviewer, you'll ask about the user's desired job role and entry level, followed by an introduction. Ask technical questions one at a time, waiting for their response. You can follow up or ask a new question after each answer. Avoid repetition and aim for depth in the conversation. Ask a maximum of 10 questions, including the introduction. Provide feedback based on their answers and areas for improvement at the end.
user:
{user_input} [/INST] </s>
"""


async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):


    config = {"max_new_tokens": 256, "temperature": 0.7, "repetition_penalty" : 1.1}

    with open('conversation_history.csv', 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        for model in MODEL_KWARGS:
            if model not in llm_chains:
                instance.placeholder_text = (
                    f"Initiating {model}, this may take a few minutes,"
                    f"or longer, depending on your internet connection."
                )
                llm = CTransformers(**MODEL_KWARGS[model], config=config)
                prompt = PromptTemplate(template=TEMPLATE, input_variables=["user_input"])
                llm_chain = LLMChain(prompt=prompt, llm=llm)
                llm_chains[model] = llm_chain

        history = [message for message in instance.objects]
        for message in history:
            if message.user == "Mistral":
                csv_writer.writerow([message.object, ""])
            else:
                csv_writer.writerow(["", message.object])

        if contents.startswith("audio:"):
            audio_data = base64.b64decode(contents.split(",")[1])
            with io.BytesIO(audio_data) as f:
                with wave.open(f, "rb") as wf:
                    audio_bytes = wf.readframes(wf.getnframes())
                    contents = audio_bytes
                    instance.send(
                        await llm_chains[model].apredict(user_input=contents),
                        user=model.title(),
                        respond=False,
                    )
        else:
            instance.send(
                await llm_chains[model].apredict(user_input=contents),
                user=model.title(),
                respond=False,
            )

# Create a SpeechToText widget for audio input
speech_input = pn.widgets.SpeechToText(name="Record Audio", service_uri = "http://localhost:5000/speech-recognition",continuous = True)



# Create a chat interface with the callback function
chat_interface = pn.chat.ChatInterface(widgets = [speech_input],callback=callback, placeholder_threshold=0.1)
chat_interface.send(
    "Send a message or record audio to get a reply from Mistral",
    user="System",
    respond=False,
)

# Combine the speech panel and chat interface into a single layout
# combined_layout = pn.Column(speech_panel, chat_interface)

# Display the combined layout
chat_interface.servable()
