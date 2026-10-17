from flask import Blueprint, render_template, request
from flask_login import login_required

from modules.text_detector import detect_text

text = Blueprint("text", __name__)


@text.route("/text", methods=["GET", "POST"])
@login_required
def text_detection():

    result = None
    submitted_text = ""

    if request.method == "POST":

        submitted_text = request.form.get("text", "").strip()

        if submitted_text:

            try:
                result = detect_text(submitted_text)

            except Exception as e:
                result = {
                    "label": "ERROR",
                    "confidence": 0,
                    "error": str(e)
                }

    return render_template(
        "text.html",
        result=result,
        submitted_text=submitted_text
    )