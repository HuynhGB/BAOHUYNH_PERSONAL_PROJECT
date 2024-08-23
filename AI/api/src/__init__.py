import os
from flask import Flask
from flask_cors import CORS
from src.apis.en.v1 import v1_en

from src.apis.no.v1 import v1_no
from src.apis.vi.v1 import v1_vi
from src.config import config
from src.apis.datetime_utils import DatetimeUtils


def create_app(app_environment=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    allowed_origins = ["https://unique-heroic-egret.ngrok-free.app"]
    CORS(app, supports_credentials=True, origins=allowed_origins)

    app.config.from_mapping(
        SECRET_KEY="dev",
        DATABASE=os.path.join(app.instance_path, "flaskr.sqlite"),
    )

    if app_environment is None:
        # load the instance config, if it exists, when not testing
        app.config.from_object(config[os.getenv("FLASK_ENV", "dev")])
    else:
        # load the test config if passed in
        app.config.from_object(config[app_environment])

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    app.register_blueprint(v1_en)
    app.register_blueprint(v1_no)
    app.register_blueprint(v1_vi)

    app.root_path

    return app
