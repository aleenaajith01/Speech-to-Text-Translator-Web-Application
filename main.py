from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile

app = FastAPI()

def transcribe_audio(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file)
        tmp_path = tmp.name

    # Load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(tmp_path)
    audio = whisper.pad_or_trim(audio)

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language: {detected_language}")

    # Decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # Print the recognized text
    print(result.text)

    return result.text, detected_language

# Load the Whisper model
model = whisper.load_model("base")


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    contents = await file.read()
    # audio = BytesIO(contents)
    transcription, detected_language = transcribe_audio(contents)
    # print(f"Returning: Transcription: {transcription}, Language: {detected_language}")
    return {"transcription": transcription, "detected_language": detected_language}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)