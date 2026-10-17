from flask import Blueprint, render_template, request
from flask_login import login_required
import os
from werkzeug.utils import secure_filename

from modules.image_detector import detect_image


image = Blueprint("image", __name__)


@image.route("/image", methods=["GET", "POST"])
@login_required
def image_detection():

    result = None
    filename = None

    if request.method == "POST":

        file = request.files.get("image")

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
                # Run SMOGY AI/Real detector
                result = detect_image(filepath)

            except Exception as e:

                result = {
                    "label": "ERROR",
                    "confidence": 0,
                    "ai_probability": 0,
                    "real_probability": 0,
                    "error": str(e)
                }

    return render_template(
        "image.html",
        result=result,
        filename=filename
    )