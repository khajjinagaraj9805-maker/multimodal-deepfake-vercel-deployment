from flask import Flask, render_template
from flask_login import LoginManager

from config import Config
from database import db
from models import User

from routes.auth import auth
from routes.dashboard import dashboard


# THESE MUST EXIST
from routes.text import text
from routes.image import image
from routes.video import video
from routes.audio import audio

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "auth.login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

app.register_blueprint(auth)
app.register_blueprint(dashboard)

# THESE MUST EXIST
app.register_blueprint(text)
app.register_blueprint(image)
app.register_blueprint(video)
app.register_blueprint(audio)

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)