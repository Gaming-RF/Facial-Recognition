from flask import Flask
import config
from app import storage


def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    storage.init_db()
    from app.routes import bp
    app.register_blueprint(bp)
    print("Face Recognition System ready.")
    return app
