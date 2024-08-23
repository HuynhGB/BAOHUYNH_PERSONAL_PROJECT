from flask import jsonify

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging

device = "cuda"
model_name = "mkhalifa/flan-t5-large-gsm8k"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache/")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="./cache/").to(
    device
)

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")


def ask_model(input_text):

    inputs = tokenizer(input_text, return_tensors="pt", return_attention_mask=False).to(
        device
    )
    outputs = model.generate(**inputs, max_length=700)
    output_text = tokenizer.batch_decode(outputs)[0]

    return output_text
