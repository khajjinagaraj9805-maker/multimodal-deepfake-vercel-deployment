import cv2
import os

from modules.image_detector import detect_image


def detect_video(video_path):

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()
        raise ValueError(
            "Video contains no readable frames."
        )

    if fps <= 0:
        fps = 30.0

    # Target approximately 12 frames per second
    target_fps = 6

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    temp_folder = os.path.join(
        project_root,
        "static",
        "video_frames"
    )

    os.makedirs(temp_folder, exist_ok=True)

    frame_results = []
    frame_number = 0
    next_sample_time = 0.0

    while True:

        success, frame = cap.read()

        if not success:
            break

        timestamp = frame_number / fps

        # Select frames based on timestamps
        if timestamp + 1e-9 >= next_sample_time:

            frame_filename = (
                f"frame_{frame_number:06d}.jpg"
            )

            frame_path = os.path.join(
                temp_folder,
                frame_filename
            )

            if not cv2.imwrite(frame_path, frame):
                print(
                    f"Could not save frame {frame_number}"
                )
                frame_number += 1
                continue

            try:
                result = detect_image(frame_path)

                frame_results.append({
                    "frame_number": frame_number,
                    "timestamp": round(timestamp, 3),
                    "frame_path": os.path.relpath(
                        frame_path,
                        project_root
                    ).replace("\\", "/"),
                    "label": result.get(
                        "label", "UNCERTAIN"
                    ),
                    "confidence": result.get(
                        "confidence", 0
                    ),
                    "ai_probability": result.get(
                        "ai_probability", 0
                    ),
                    "real_probability": result.get(
                        "real_probability", 0
                    )
                })

            except Exception as e:
                print(
                    f"Frame {frame_number} failed: {e}"
                )

            # Advance the sampling schedule even if inference fails
            next_sample_time += 1.0 / target_fps

        frame_number += 1

    cap.release()

    if not frame_results:
        raise ValueError(
            "Could not analyze any video frames."
        )

    # Calculate overall probabilities
    ai_scores = [
        frame["ai_probability"]
        for frame in frame_results
    ]

    real_scores = [
        frame["real_probability"]
        for frame in frame_results
    ]

    average_ai = sum(ai_scores) / len(ai_scores)
    average_real = sum(real_scores) / len(real_scores)

    if average_ai >= average_real:
        overall_label = "AI GENERATED"
        overall_confidence = average_ai
    else:
        overall_label = "REAL VIDEO"
        overall_confidence = average_real

    return {
        "label": overall_label,
        "confidence": round(overall_confidence, 2),
        "ai_probability": round(average_ai, 2),
        "real_probability": round(average_real, 2),
        "frames_analyzed": len(frame_results),
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "frames": frame_results
    }
