
import os
import threading

import numpy as np
import soundfile as sf
import onnxruntime as ort


# ============================================================
# DHWANI AUDIO DEEPFAKE DETECTOR
# ============================================================

MODEL_NAME = "Dhwani"
MODEL_SOURCE = "Local ONNX model"
MODEL_FORMAT = "ONNX"
MODEL_FILENAME = "best_model.onnx"
INFERENCE_DEVICE = "CPU"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 4
STRIDE_SECONDS = 2
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
STRIDE_SAMPLES = SAMPLE_RATE * STRIDE_SECONDS
TEMPERATURE = 1.362

# Resolve the project root from the modules directory.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "audio_model_lab",
    "dhwani",
    MODEL_FILENAME,
)

print("=" * 60)
print("AUDIO DEEPFAKE DETECTOR")
print(f"Model name       : {MODEL_NAME}")
print(f"Model source     : {MODEL_SOURCE}")
print(f"Model format     : {MODEL_FORMAT}")
print(f"Model filename   : {MODEL_FILENAME}")
print(f"Inference device : {INFERENCE_DEVICE}")
print(f"Model path       : {MODEL_PATH}")
print("=" * 60)


# ============================================================
# ONNX MODEL SESSION
# Load the large model only once per Python process.
# ============================================================

_session = None
_session_lock = threading.Lock()


def _get_session():
    """Load and cache the Dhwani ONNX inference session."""
    global _session

    if _session is None:
        with _session_lock:
            if _session is None:
                if not os.path.isfile(MODEL_PATH):
                    raise FileNotFoundError(
                        f"{MODEL_NAME} ONNX model not found: "
                        f"{MODEL_PATH}"
                    )

                print(f"Loading {MODEL_NAME} audio detector...")

                options = ort.SessionOptions()

                session = ort.InferenceSession(
                    MODEL_PATH,
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )

                # Read actual ONNX input/output names instead of
                # assuming the model uses "input" and "output".
                input_names = [
                    item.name for item in session.get_inputs()
                ]
                output_names = [
                    item.name for item in session.get_outputs()
                ]

                if not input_names or not output_names:
                    raise RuntimeError(
                        "The ONNX model has no usable input/output."
                    )

                _session = session

                print(
                    f"{MODEL_NAME} detector loaded successfully!"
                )
                print(f"ONNX input names : {input_names}")
                print(f"ONNX output names: {output_names}")
                print(f"Active provider  : {session.get_providers()[0]}")

    return _session


# ============================================================
# PROBABILITY CALCULATION
# ============================================================

def _softmax(logits):
    """Convert model logits into probabilities."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)

    if logits.size < 2:
        raise ValueError(
            f"Expected at least two output classes; "
            f"received {logits.size}."
        )

    if not np.isfinite(logits).all():
        raise ValueError("The model returned invalid output values.")

    logits = logits / TEMPERATURE
    logits = logits - np.max(logits)

    exp_values = np.exp(logits)
    total = np.sum(exp_values)

    if not np.isfinite(total) or total <= 0:
        raise ValueError("Could not calculate model probabilities.")

    return exp_values / total


# ============================================================
# AUDIO DEEPFAKE DETECTION
# ============================================================

def detect_dhwani(audio_path):
    """
    Analyze audio with Dhwani.

    Returns the fields expected by the existing audio.html:
    label, confidence, ai_probability, real_probability.

    Class index 1 is treated as the AI-generated class,
    matching the existing detector configuration.
    """
    try:
        session = _get_session()

        # Read audio as float32.
        audio, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=False,
        )

        # Convert stereo audio to mono.
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        if audio.ndim != 1:
            raise ValueError(
                "Unsupported audio shape. Expected mono or stereo audio."
            )

        if audio.size == 0:
            raise ValueError("The uploaded audio file is empty.")

        if not np.isfinite(audio).all():
            raise ValueError(
                "The audio contains invalid numeric values."
            )

        if np.max(np.abs(audio)) < 1e-6:
            raise ValueError(
                "The audio is silent or too quiet to analyze."
            )

        # Resample audio to the required 16 kHz.
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=SAMPLE_RATE,
            ).astype(np.float32)

        if audio.size == 0:
            raise ValueError(
                "No audio samples remain after resampling."
            )

        # Pad short audio to one complete four-second window.
        if audio.size < WINDOW_SAMPLES:
            audio = np.pad(
                audio,
                (0, WINDOW_SAMPLES - audio.size),
                mode="constant",
            )

        # Use four-second windows with a two-second stride.
        starts = range(
            0,
            audio.size - WINDOW_SAMPLES + 1,
            STRIDE_SAMPLES,
        )

        fake_probabilities = []

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        for start in starts:
            window = audio[
                start:start + WINDOW_SAMPLES
            ]

            model_input = window.astype(
                np.float32
            )[None, :]

            outputs = session.run(
                [output_name],
                {input_name: model_input},
            )[0]

            # Expected output: batch dimension followed by class logits.
            logits = np.asarray(outputs)

            if logits.ndim == 2:
                logits = logits[0]
            elif logits.ndim != 1:
                raise ValueError(
                    "Unexpected ONNX output shape: "
                    f"{logits.shape}"
                )

            probabilities = _softmax(logits)

            # Preserve the existing configuration:
            # output index 1 represents AI-generated audio.
            fake_probabilities.append(
                float(probabilities[1])
            )

        if not fake_probabilities:
            raise ValueError(
                "Could not create an audio analysis window."
            )

        # Preserve the existing maximum-window strategy.
        fake_probability = max(fake_probabilities) * 100.0
        real_probability = 100.0 - fake_probability

        if fake_probability >= real_probability:
            label = "AI GENERATED AUDIO"
            confidence = fake_probability
        else:
            label = "REAL AUDIO"
            confidence = real_probability

        result = {
            "label": label,
            "confidence": round(confidence, 2),
            "ai_probability": round(fake_probability, 2),
            "real_probability": round(real_probability, 2),
        }

        print(
            f"[{MODEL_NAME}] Prediction: {label} | "
            f"Confidence: {confidence:.2f}%"
        )

        return result

    except Exception as exc:
        print(f"[{MODEL_NAME}] Detection error: {exc}")

        return {
            "label": "ERROR",
            "confidence": 0.0,
            "ai_probability": 0.0,
            "real_probability": 0.0,
            "error": str(exc),
        }
