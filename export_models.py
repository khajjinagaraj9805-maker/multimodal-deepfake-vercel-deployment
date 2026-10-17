from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoImageProcessor,
    AutoModelForImageClassification,
)

text_id = "frankhanhj/roberta-base-all-sources-ai-identifier"
image_id = "Smogy/SMOGY-Ai-images-detector"

print("Exporting text model...")
AutoTokenizer.from_pretrained(
    text_id, local_files_only=True
).save_pretrained("models/text_detector")

AutoModelForSequenceClassification.from_pretrained(
    text_id, local_files_only=True
).save_pretrained("models/text_detector", safe_serialization=True)

print("Exporting image model...")
AutoImageProcessor.from_pretrained(
    image_id, local_files_only=True
).save_pretrained("models/image_detector")

AutoModelForImageClassification.from_pretrained(
    image_id, local_files_only=True
).save_pretrained("models/image_detector", safe_serialization=True)

print("SUCCESS: Both models exported locally.")
