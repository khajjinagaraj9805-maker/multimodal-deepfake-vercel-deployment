
import os
import traceback

PROJECT = r"C:\MULTI MODAL DEEP FAKE DETECTION PROJECT"

# Update these two paths if you have suitable test media.
IMAGE_PATH = os.path.join(
    PROJECT, "static", "uploads", "testai.jpeg"
)

VIDEO_PATH = os.path.join(
    PROJECT, "static", "uploads", "testvedio2.mp4"
)
AUDIO_PATH = os.path.join(
    PROJECT,
    "audio_model_lab",
    "test_audio_16k",
    "testaudio10_16k.wav"
)


def run_test(name, function):
    print(f"\n{'=' * 18} {name} {'=' * 18}")
    try:
        result = function()
        print("TEST PASSED: Function returned a result.")
        print("Result:", result)
    except Exception as exc:
        print(f"TEST FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)


def test_text():
    from modules.text_detector import detect_text

    sample = (
        "Artificial intelligence can help students learn new concepts, "
        "but they should verify information and think critically."
    )
    return detect_text(sample)


def test_image():
    if not IMAGE_PATH or not os.path.isfile(IMAGE_PATH):
        return "SKIPPED: Set IMAGE_PATH to an existing image file."

    from modules.image_detector import detect_image
    return detect_image(IMAGE_PATH)


def test_audio():
    if not os.path.isfile(AUDIO_PATH):
        return f"SKIPPED: Audio file not found: {AUDIO_PATH}"

    from modules.audio_detector import detect_dhwani
    return detect_dhwani(AUDIO_PATH)


def test_video():
    if not VIDEO_PATH or not os.path.isfile(VIDEO_PATH):
        return "SKIPPED: Set VIDEO_PATH to an existing video file."

    from modules.video_detector import detect_video
    return detect_video(VIDEO_PATH)


if __name__ == "__main__":
    print("MULTIMODAL DEEPFAKE DETECTOR TEST")
    print("Project:", PROJECT)

    run_test("TEXT", test_text)
    run_test("IMAGE", test_image)
    run_test("AUDIO", test_audio)
    run_test("VIDEO", test_video)

    print("\nAll test attempts finished.")
