
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_NAME = os.path.join(
    PROJECT_ROOT, "models", "text_detector"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Loading local text detector...")
print("Device:", device)
print("Model path:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

model.to(device)
model.eval()

print("Local text detector loaded successfully!")
print("Model labels:", model.config.id2label)


def detect_text(text):
    if not text or not text.strip():
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "ai_probability": 0.0,
            "human_probability": 0.0,
        }

    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits, dim=-1
    )[0]

    labels = model.config.id2label

    ai_index = next(
        (
            int(index)
            for index, name in labels.items()
            if "ai" in name.lower()
            or "artificial" in name.lower()
        ),
        None,
    )

    human_index = next(
        (
            int(index)
            for index, name in labels.items()
            if "human" in name.lower()
        ),
        None,
    )

    if ai_index is None or human_index is None:
        raise ValueError(
            f"Unexpected model labels: {labels}"
        )

    ai_probability = probabilities[ai_index].item() * 100
    human_probability = (
        probabilities[human_index].item() * 100
    )

    predicted_index = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_index].item() * 100

    if predicted_index == ai_index:
        label = "AI GENERATED"
    else:
        label = "HUMAN GENERATED"

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "ai_probability": round(ai_probability, 2),
        "human_probability": round(human_probability, 2),
    }
