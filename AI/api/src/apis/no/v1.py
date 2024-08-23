from flask import jsonify, request
from flask_cors import cross_origin
from src.models.faster_whisper import faster_whisper_recognize

from flask import jsonify, request
from flask_cors import cross_origin

from flask import Blueprint

from src.models.no_bert import No_Command_Analyzer

v1_no = Blueprint("v1_no", __name__)


@v1_no.route("/v1/no/recognize", methods=["POST"])
@cross_origin()
def v2_en_recognize():
    req_data = request.get_json()

    file_data = req_data["file_data"]

    output_text = faster_whisper_recognize(file_data)
    return jsonify(output_text=output_text)


@v1_no.route("/v1/no/analyze", methods=["POST"])
@cross_origin()
def v1_en_analyze():
    req_data = request.get_json()

    cmd = req_data["command"]
    ca = No_Command_Analyzer()
    output_cmd = ca.analyze(cmd)
    print(f"Analyzed command: {output_cmd}")

    return jsonify(output_cmd)


@v1_no.route("/v1/no/train", methods=["POST"])
@cross_origin()
def v1_en_train():
    req_data = request.get_json()

    train_data = req_data["train_data"]
    ca = No_Command_Analyzer()
    is_trained = ca.train(train_data)

    is_trained = True
    return jsonify(is_trained=is_trained)
