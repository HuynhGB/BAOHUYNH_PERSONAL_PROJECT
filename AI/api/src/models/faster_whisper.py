from faster_whisper import WhisperModel
import torch
from transformers import logging
import base64
import os
import io
import uuid

LANG_ID = "en"
MODEL_ID = "./models/nb-whisper-large-beta"
SAMPLES = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Run on GPU with FP16
model = WhisperModel(MODEL_ID, device="cuda", compute_type="int8")  # int8, float32

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")


def faster_whisper_recognize(base64_encoded_sound_file_data, lang_code="no"):

    sound_file_data = base64.b64decode(base64_encoded_sound_file_data)
    # assert sound_file_data.startswith(b'wav')  # just to prove it is an Ogg Vorbis file
    sound_file = io.BytesIO(sound_file_data)
    temporarylocation = f"./audio/{uuid.uuid4().hex}.wav"
    with open(temporarylocation, "wb") as out:  ## Open temporary file as bytes
        out.write(sound_file.read())  ## Read bytes into file

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        temporarylocation, beam_size=5, language=lang_code
    )

    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )

    segments = list(segments)
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    ## Do stuff with module/file
    os.remove(temporarylocation)  ## Delete file when done
    output_text = "".join(o.text for o in segments)
    return output_text
