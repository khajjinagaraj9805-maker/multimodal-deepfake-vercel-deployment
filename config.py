
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load local environment variables only outside Vercel.
if os.environ.get("VERCEL") != "1":
    load_dotenv()


class Config:
    IS_VERCEL = os.environ.get("VERCEL") == "1"

    SECRET_KEY = os.environ.get("SECRET_KEY")
    MYSQL_HOST = os.environ.get("MYSQL_HOST")
    MYSQL_USER = os.environ.get("MYSQL_USER")
    MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
    MYSQL_DB = os.environ.get("MYSQL_DB")

    if not IS_VERCEL:
        SECRET_KEY = SECRET_KEY or "local-development-only-change-me"
        MYSQL_HOST = MYSQL_HOST or "localhost"
        MYSQL_USER = MYSQL_USER or "root"
        MYSQL_PASSWORD = MYSQL_PASSWORD or ""
        MYSQL_DB = MYSQL_DB or "deepfake_detection"

    if IS_VERCEL and not all(
        [SECRET_KEY, MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB]
    ):
        raise RuntimeError(
            "Set SECRET_KEY, MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD "
            "and MYSQL_DB in the Vercel environment variables."
        )

    user = quote_plus(MYSQL_USER)
    password = quote_plus(MYSQL_PASSWORD)

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{user}:{password}@{MYSQL_HOST}/{MYSQL_DB}"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join("static", "uploads")
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024
