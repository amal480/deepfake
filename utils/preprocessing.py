import base64
import io
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000
CHUNK_SEC = 4
CHUNK_LEN = CHUNK_SEC * TARGET_SR
OVERLAP_SEC = 2
OVERLAP_LEN = OVERLAP_SEC * TARGET_SR


def decode_audio(base64_audio: str):
    audio_bytes = base64.b64decode(base64_audio)
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

    # OPTIONAL SAFETY (recommended)
    wav = np.clip(wav, -1.0, 1.0)

    return wav


def chunk_audio(wav: np.ndarray):
    """
    Returns:
    - list of chunks if audio is long
    - single padded chunk if audio is short
    """
    if len(wav) <= CHUNK_LEN:
        return [np.pad(wav, (0, CHUNK_LEN - len(wav)))]

    chunks = []
    step = CHUNK_LEN - OVERLAP_LEN

    for start in range(0, len(wav) - CHUNK_LEN + 1, step):
        chunks.append(wav[start:start + CHUNK_LEN])

    return chunks