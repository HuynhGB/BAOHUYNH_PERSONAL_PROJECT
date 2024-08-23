import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, logging, pipeline

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache/")
model = AutoModelForQuestionAnswering.from_pretrained(
    model_name, cache_dir="./cache/"
).to("cuda")
question_answerer = pipeline(
    "question-answering", model=model_name, cache_dir="./cache/"
)

logging.set_verbosity_info()
logger = logging.get_logger("transformers")


def answer(question, context):
    logger.info("INFO: Question    : {0}".format(question))

    f = open("./data/context/company_info.txt")
    context = f.read()
    f.close()

    inputs = tokenizer(
        question, context, truncation="only_second", return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[
        0, answer_start_index : answer_end_index + 1
    ]
    answer = tokenizer.decode(predict_answer_tokens)

    test = question_answerer(question=question, context=context)

    if test["score"] < 0.01:
        test["answer"] = "Sorry, I don't have any information for your question."
    else:
        logger.info(f'INFO: Answer      : {test["answer"]}')

    return test["answer"]
