import os
import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from models.cnn_baseline import AIVoiceCNN
from utils.preprocessing import decode_audio, chunk_audio
from utils.explanation import generate_explanation


torch.set_grad_enabled(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ================= CONFIG =================
API_KEY = os.getenv("API_KEY", "CHANGE_ME")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 16000

# ================= LOAD MODELS =================
ai_model = AIVoiceCNN().to(DEVICE)
ai_model.load_state_dict(
    torch.load("models/best_model.pt", map_location=DEVICE)
)
ai_model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=320,
    n_mels=128
).to(DEVICE)

db_transform = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

# ================= API =================
app = FastAPI(title="AI Voice Detection API")

class AudioRequest(BaseModel):
    audioBase64: str


@app.post("/api/voice-detection")
def detect_voice(
    request: AudioRequest,
    x_api_key: str = Header(None)
):
    # -------- AUTH --------
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # -------- DECODE --------
    wav = decode_audio(request.audioBase64)
    chunks = chunk_audio(wav)

    aggregated = len(chunks) > 1
    ai_probs = []

    with torch.inference_mode():
        for chunk in chunks:
            wav_tensor = torch.from_numpy(chunk).float()

            # ---- MEL FEATURE ----
            mel = db_transform(mel_transform(wav_tensor))

            # ===== AI CNN =====
            ai_mel = mel.unsqueeze(0).unsqueeze(0)  # (1,1,128,T)
            ai_logit = ai_model(ai_mel)
            ai_prob = torch.sigmoid(ai_logit).item()
            ai_probs.append(ai_prob)

    # -------- AI AGGREGATION --------
    final_prob = float(np.mean(ai_probs))

    if final_prob >= 0.5:
        classification = "AI_GENERATED"
        confidence_score = final_prob
    else:
        classification = "HUMAN"
        confidence_score = 1.0 - final_prob

    # -------- RESPONSE --------
    return {
        "status": "success",
        "classification": classification,
        "confidenceScore": round(confidence_score, 4),
        "explanation": generate_explanation(classification, aggregated)
    }

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok"}