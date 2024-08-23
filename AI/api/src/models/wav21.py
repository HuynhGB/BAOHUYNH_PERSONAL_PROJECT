import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, logging
import torchaudio
import base64
import os
import io
import uuid
import numpy

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v3"  # "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    cache_dir="./cache/",
)

model.to(device)

processor = AutoProcessor.from_pretrained(model_id, cache_dir="./cache/")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")


def openai_recognize(base64_encoded_sound_file_data):

    sound_file_data = base64.b64decode(base64_encoded_sound_file_data)
    # assert sound_file_data.startswith(b'wav')  # just to prove it is an Ogg Vorbis file
    sound_file = io.BytesIO(sound_file_data)
    temporarylocation = f"./audio/{uuid.uuid4().hex}.wav"
    with open(temporarylocation, "wb") as out:  ## Open temporary file as bytes
        out.write(sound_file.read())  ## Read bytes into file

    audio, sample_rate = torchaudio.load(temporarylocation)

    x = numpy.asarray(audio[0]).astype("float32")
    output_text = pipe(
        x, generate_kwargs={"task": "transcribe", "language": "en"}, batch_size=8
    )
    print(output_text)

    ## Do stuff with module/file
    os.remove(temporarylocation)  ## Delete file when done

    return output_text["text"]


def openai_recognize1():
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from datasets import load_dataset

    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    sample = ds[0]["audio"]
    input_features = processor(
        sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
    ).input_features

    # set the forced ids
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="french", task="transcribe"
    )

    # generate token ids
    predicted_ids = model.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
    print(transcription)
