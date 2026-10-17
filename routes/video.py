from flask import Blueprint, render_template, request
from flask_login import login_required
import os
from werkzeug.utils import secure_filename

from modules.video_detector import detect_video


video = Blueprint("video", __name__)


@video.route("/video", methods=["GET", "POST"])
@login_required
def video_detection():

    result = None

    if request.method == "POST":

        file = request.files.get("video")

        if file and file.filename:

            filename = secure_filename(file.filename)

            upload_folder = "static/uploads"
            os.makedirs(upload_folder, exist_ok=True)

            filepath = os.path.join(
                upload_folder,
                filename
            )

            file.save(filepath)

            try:

                result = detect_video(filepath)

            except Exception as e:

                result = {
                    "label": "ERROR",
                    "confidence": 0,
                    "error": str(e)
                }

    return render_template(
        "video.html",
        result=result
    )