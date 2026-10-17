from flask import Blueprint, render_template, request
from flask_login import login_required
import os
from werkzeug.utils import secure_filename

from modules.audio_detector import detect_dhwani


audio = Blueprint("audio", __name__)


@audio.route("/audio", methods=["GET", "POST"])
@login_required
def audio_detection():

    result = None

    if request.method == "POST":

        file = request.files.get("audio")

        if file and file.filename:

            filename = secure_filename(file.filename)

            upload_folder = "static/uploads"
            os.makedirs(upload_folder, exist_ok=True)

            filepath = os.path.join(upload_folder, filename)

            file.save(filepath)

            # Run AI audio deepfake detection
            result = detect_dhwani(filepath)

    return render_template(
        "audio.html",
        result=result
    )

