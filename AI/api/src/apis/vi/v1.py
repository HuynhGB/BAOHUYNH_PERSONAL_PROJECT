from flask import jsonify, request
from flask_cors import cross_origin
from src.models.vi_wav2vec import recognize
from src.models.vi_bert import Vi_Command_Analyzer
from flask import jsonify, request
from flask_cors import cross_origin
from flask import Blueprint

v1_vi = Blueprint("v1_vi", __name__)


@v1_vi.route("/v1/vi/recognize", methods=["POST"])
@cross_origin()
def v1_en_recognize():
    req_data = request.get_json()

    question = req_data["file_data"]

    output_text = recognize(question)
    return jsonify(output_text=output_text)


@v1_vi.route("/v1/vi/analyze", methods=["POST"])
@cross_origin()
def v1_en_analyze():
    req_data = request.get_json()

    cmd = req_data["command"]
    ca = Vi_Command_Analyzer()
    output_cmd = ca.analyze(cmd)
    print(f"Analyzed command: {output_cmd}")

    return jsonify(output_cmd)


@v1_vi.route("/v1/vi/train", methods=["POST"])
@cross_origin()
def v1_en_train():
    req_data = request.get_json()

    train_data = req_data["train_data"]
    ca = Vi_Command_Analyzer()
    is_trained = ca.train(train_data)
    return jsonify(is_trained=is_trained)
