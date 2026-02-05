import os
import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from models.cnn_baseline import AIVoiceCNN
from models.lid_cnn_best3 import LIDCNN
from utils.preprocessing import decode_audio, chunk_audio
from utils.explanation import generate_explanation

# ================= CONFIG =================
API_KEY = os.getenv("API_KEY", "CHANGE_ME")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 16000
TARGET_FRAMES = 200

IDX_TO_LANG = {
    0: "english",
    1: "hindi",
    2: "malayalam",
    3: "tamil",
    4: "telugu"
}

# ================= LOAD MODELS =================
ai_model = AIVoiceCNN().to(DEVICE)
ai_model.load_state_dict(
    torch.load("models/best_model.pt", map_location=DEVICE)
)
ai_model.eval()

lid_model = LIDCNN(num_classes=5).to(DEVICE)
lid_model.load_state_dict(
    torch.load("models/lid_cnn_best3.pt", map_location=DEVICE)
)
lid_model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=320,
    n_mels=128
).to(DEVICE)

db_transform = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

# ================= HELPERS =================
def fix_mel_length(mel):
    """
    mel: (1, 128, T) → (1, 128, TARGET_FRAMES)
    """
    T = mel.shape[-1]

    if T > TARGET_FRAMES:
        mel = mel[..., :TARGET_FRAMES]
    elif T < TARGET_FRAMES:
        mel = F.pad(mel, (0, TARGET_FRAMES - T))

    return mel

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
    lid_preds = []

    with torch.no_grad():
        for chunk in chunks:
            wav_tensor = torch.from_numpy(chunk).float().to(DEVICE)

            # ---- MEL FEATURE ----
            mel = mel_transform(wav_tensor)
            mel = db_transform(mel)

            # ===== AI CNN =====
            ai_mel = mel.unsqueeze(0).unsqueeze(0)  # (1,1,128,T)
            ai_logit = ai_model(ai_mel)
            ai_prob = torch.sigmoid(ai_logit).item()
            ai_probs.append(ai_prob)

            # ===== LID CNN =====
            lid_mel = mel.unsqueeze(0)              # (1,128,T)
            lid_mel = fix_mel_length(lid_mel)       # (1,128,200)
            lid_mel = lid_mel.unsqueeze(0)          # (1,1,128,200)

            lid_logits = lid_model(lid_mel)
            lid_pred = torch.argmax(lid_logits, dim=1).item()
            lid_preds.append(lid_pred)

    # -------- AI AGGREGATION --------
    final_prob = float(np.mean(ai_probs))

    if final_prob >= 0.5:
        classification = "AI_GENERATED"
        confidence_score = final_prob
    else:
        classification = "HUMAN"
        confidence_score = 1.0 - final_prob

    # -------- LANGUAGE MAJORITY VOTE --------
    lid_preds = np.array(lid_preds)
    voted_idx = int(np.bincount(lid_preds).argmax())
    detected_language = IDX_TO_LANG[voted_idx]

    # -------- RESPONSE (SPEC COMPLIANT) --------
    return {
        "status": "success",
        "language": detected_language.capitalize(),
        "classification": classification,
        "confidenceScore": round(confidence_score, 4),
        "explanation": generate_explanation(classification, aggregated)
    }

@app.get("/")
def root():
    return {"status": "ok"}