import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, logging
import torchaudio
import base64
import os
import io
import time

LANG_ID = "en"
MODEL_ID = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
SAMPLES = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, cache_dir="./cache/")
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, cache_dir="./cache/").to(device)


logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")


def recognize(base64_encoded_sound_file_data):

    sound_file_data = base64.b64decode(base64_encoded_sound_file_data)
    # assert sound_file_data.startswith(b'wav')  # just to prove it is an Ogg Vorbis file
    sound_file = io.BytesIO(sound_file_data)
    temporarylocation = f"./audio/{round(time.time() * 1000)}.wav"
    with open(temporarylocation, "wb") as out:  ## Open temporary file as bytes
        out.write(sound_file.read())  ## Read bytes into file

    audio, sample_rate = torchaudio.load(temporarylocation)
    input_data = processor.feature_extractor(
        audio[0], sampling_rate=sample_rate, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model(**input_data)

    output_text = processor.tokenizer.decode(
        output.logits.argmax(dim=-1)[0].detach().cpu().numpy()
    )
    print(output_text)

    ## Do stuff with module/file
    os.remove(temporarylocation)  ## Delete file when done

    return output_text
