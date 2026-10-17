
import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"

print("Loading AI audio deepfake detector...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

ID2LABEL = {
    int(key): str(value).lower()
    for key, value in model.config.id2label.items()
}

print("Audio detector loaded successfully!")
print("Labels:", ID2LABEL)


def detect_audio(audio_path):
    try:
        audio, sample_rate = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        if audio.size == 0:
            raise ValueError("Audio file contains no audio data.")

        if not np.isfinite(audio).all():
            raise ValueError("Audio contains invalid numeric values.")

        if np.max(np.abs(audio)) < 1e-6:
            raise ValueError("Audio is silent or too quiet.")

        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)
            probabilities = torch.softmax(
                outputs.logits, dim=-1
            )[0].cpu().numpy()

        class_probabilities = {
            ID2LABEL[i]: float(probabilities[i]) * 100
            for i in range(len(probabilities))
        }

        if "real" not in class_probabilities or "fake" not in class_probabilities:
            raise ValueError(
                f"Unexpected model labels: {ID2LABEL}"
            )

        real_probability = class_probabilities["real"]
        fake_probability = class_probabilities["fake"]

        if fake_probability >= real_probability:
            label = "AI GENERATED AUDIO"
            confidence = fake_probability
        else:
            label = "REAL AUDIO"
            confidence = real_probability

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "ai_probability": round(fake_probability, 2),
            "real_probability": round(real_probability, 2)
        }

    except Exception as e:
        return {
            "label": "ERROR",
            "confidence": 0.0,
            "ai_probability": 0.0,
            "real_probability": 0.0,
            "error": str(e)
        }
