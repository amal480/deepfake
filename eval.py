import os
import time
import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from models.cnn_baseline import AIVoiceCNN
from utils.preprocessing import chunk_audio


# ================= CONFIG =================
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE    = 16000
THRESHOLD      = 0.5

DATASET_ROOT   = "dev_test"          # folder containing clips/ and clips_AI/
HUMAN_DIR      = os.path.join(DATASET_ROOT, "clips")
AI_DIR         = os.path.join(DATASET_ROOT, "clips_AI")

SUPPORTED_EXT  = {".wav"}

# ================= LOAD MODEL =================
print(f"[INFO] Loading model on {DEVICE} ...")
model = AIVoiceCNN().to(DEVICE)
model.load_state_dict(torch.load("models/best_model.pt", map_location=DEVICE))
model.eval()

mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=320,
    n_mels=128
).to(DEVICE)

db_transform = T.AmplitudeToDB().to(DEVICE)


# ================= HELPERS =================
def load_audio(path: str) -> np.ndarray:
    """Load audio file → resampled mono numpy array."""
    waveform, sr = torchaudio.load(path)

    # mix down to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample if needed
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    return waveform.squeeze(0).numpy()          # (T,)


def predict(wav: np.ndarray) -> tuple[float, float]:
    """
    Run inference on a waveform.
    Returns (ai_probability, elapsed_seconds).
    """
    chunks    = chunk_audio(wav)
    ai_probs  = []

    t_start = time.perf_counter()

    with torch.inference_mode():
        for chunk in chunks:
            wav_t  = torch.from_numpy(chunk).float()
            mel    = db_transform(mel_transform(wav_t))
            ai_mel = mel.unsqueeze(0).unsqueeze(0)      # (1,1,128,T)
            logit  = model(ai_mel)
            prob   = torch.sigmoid(logit).item()
            ai_probs.append(prob)

    elapsed = time.perf_counter() - t_start
    return float(np.mean(ai_probs)), elapsed


def collect_files(folder: str) -> list[str]:
    return [
        str(p) for p in Path(folder).rglob("*")
        if p.suffix.lower() in SUPPORTED_EXT
    ]


# ================= EVALUATE =================
def evaluate():
    human_files = collect_files(HUMAN_DIR)
    ai_files    = collect_files(AI_DIR)

    print(f"[INFO] Found {len(human_files)} human clips, {len(ai_files)} AI clips\n")

    y_true, y_pred, latencies = [], [], []
    errors = []

    def run_split(files, true_label, label_name):
        for path in files:
            try:
                wav           = load_audio(path)
                prob, elapsed = predict(wav)
                pred          = 1 if prob >= THRESHOLD else 0

                y_true.append(true_label)
                y_pred.append(pred)
                latencies.append(elapsed)

                status = "✓" if pred == true_label else "✗"
                print(f"  [{status}] {os.path.basename(path):<40} "
                      f"prob={prob:.4f}  latency={elapsed*1000:.1f}ms")

            except Exception as e:
                errors.append((path, str(e)))
                print(f"  [!] {os.path.basename(path)} — ERROR: {e}")

    print("── HUMAN clips (label=0) ──────────────────────────────────────")
    run_split(human_files, true_label=0, label_name="HUMAN")

    print("\n── AI clips (label=1) ────────────────────────────────────────")
    run_split(ai_files,    true_label=1, label_name="AI_GENERATED")

    if not y_true:
        print("[ERROR] No files were processed.")
        return

    # ================= METRICS =================
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    lat    = np.array(latencies) * 1000        # → ms

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print("\n" + "═" * 55)
    print("  EVALUATION RESULTS")
    print("═" * 55)
    print(f"  Total samples   : {len(y_true)}  "
          f"(Human: {(y_true==0).sum()}  |  AI: {(y_true==1).sum()})")
    print(f"  Errors skipped  : {len(errors)}")
    print("─" * 55)
    print(f"  Accuracy        : {acc*100:.2f}%")
    print(f"  Precision       : {prec*100:.2f}%")
    print(f"  Recall          : {rec*100:.2f}%")
    print(f"  F1 Score        : {f1*100:.2f}%")
    print("─" * 55)
    print("  Confusion Matrix  (rows=actual, cols=predicted)")
    print(f"                  Pred HUMAN   Pred AI")
    print(f"  Actual HUMAN  :  {cm[0,0]:>6}       {cm[0,1]:>6}")
    print(f"  Actual AI     :  {cm[1,0]:>6}       {cm[1,1]:>6}")
    print("─" * 55)
    print("  Latency per file (inference only)")
    print(f"    Mean          : {lat.mean():.1f} ms")
    print(f"    Median        : {np.median(lat):.1f} ms")
    print(f"    Std           : {lat.std():.1f} ms")
    print(f"    Min / Max     : {lat.min():.1f} ms / {lat.max():.1f} ms")
    print(f"    p95           : {np.percentile(lat, 95):.1f} ms")
    print("─" * 55)
    print("\n  Per-class Report:")
    print(classification_report(y_true, y_pred, target_names=["HUMAN", "AI_GENERATED"]))

    if errors:
        print("\n  Files with errors:")
        for p, e in errors:
            print(f"    {p}: {e}")


if __name__ == "__main__":
    evaluate()