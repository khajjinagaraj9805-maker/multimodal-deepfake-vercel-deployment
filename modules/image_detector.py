
import os
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_NAME = os.path.join(
    PROJECT_ROOT, "models", "image_detector"
)

print("Loading local SMOGY AI/Real image detector...")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)
print("Model path:", MODEL_NAME)

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

model.to(device)
model.eval()

print("Local SMOGY model loaded successfully!")
print("Labels:", model.config.id2label)


def detect_image(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    labels = model.config.id2label

    ai_index = next(
        (
            int(index)
            for index, name in labels.items()
            if "artificial" in name.lower()
            or "ai" in name.lower()
        ),
        None,
    )

    real_index = next(
        (
            int(index)
            for index, name in labels.items()
            if "human" in name.lower()
        ),
        None,
    )

    if ai_index is None or real_index is None:
        raise ValueError(
            f"Unexpected image model labels: {labels}"
        )

    ai_probability = probabilities[ai_index].item() * 100
    real_probability = probabilities[real_index].item() * 100

    predicted_class = torch.argmax(probabilities).item()
    raw_label = labels[predicted_class]

    if predicted_class == ai_index:
        label = "AI GENERATED"
    elif predicted_class == real_index:
        label = "REAL IMAGE"
    else:
        label = "UNCERTAIN"

    confidence = probabilities[predicted_class].item() * 100

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "ai_probability": round(ai_probability, 2),
        "real_probability": round(real_probability, 2),
        "raw_label": raw_label,
    }

#for video frames batchwise processing and doesn't effect existing image detection
def detect_images(image_paths, batch_size=8):
    """
    Batch inference for video frames only.
    Existing detect_image() remains unchanged.
    """
    results = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        images = []

        try:
            for image_path in batch_paths:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))

            inputs = processor(
                images=images,
                return_tensors="pt"
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = model(**inputs)

            batch_probabilities = torch.softmax(
                outputs.logits, dim=-1
            )

            labels = model.config.id2label

            ai_index = next(
                (
                    int(index)
                    for index, name in labels.items()
                    if "artificial" in name.lower()
                    or "ai" in name.lower()
                ),
                None
            )

            real_index = next(
                (
                    int(index)
                    for index, name in labels.items()
                    if "human" in name.lower()
                ),
                None
            )

            if ai_index is None or real_index is None:
                raise ValueError(
                    f"Unexpected image model labels: {labels}"
                )

            for probabilities in batch_probabilities:
                ai_probability = (
                    probabilities[ai_index].item() * 100
                )
                real_probability = (
                    probabilities[real_index].item() * 100
                )

                predicted_class = torch.argmax(
                    probabilities
                ).item()

                raw_label = labels[predicted_class]

                if predicted_class == ai_index:
                    label = "AI GENERATED"
                elif predicted_class == real_index:
                    label = "REAL IMAGE"
                else:
                    label = "UNCERTAIN"

                confidence = (
                    probabilities[predicted_class].item() * 100
                )

                results.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "ai_probability": round(
                        ai_probability, 2
                    ),
                    "real_probability": round(
                        real_probability, 2
                    ),
                    "raw_label": raw_label
                })

        except Exception as e:
            print(
                f"Batch inference failed: {e}. "
                "Falling back to individual inference."
            )

            for image_path in batch_paths:
                try:
                    results.append(detect_image(image_path))
                except Exception as image_error:
                    print(
                        f"Image inference failed for "
                        f"{image_path}: {image_error}"
                    )
                    results.append(None)

        finally:
            for image in images:
                image.close()

    return results

