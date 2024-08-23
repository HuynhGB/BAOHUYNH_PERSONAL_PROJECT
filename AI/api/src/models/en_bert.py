from src.models.data_maker import NERDataMaker
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    logging,
    pipeline,
)
from py_linq import Enumerable
from src.models.no_bert import CommandAnalyer
from src.stringutils import StringUtils
from flask import current_app as app
import os.path


class En_Command_Analyzer(CommandAnalyer):
    out_dir = "./models/distilbert-base-uncased-fine-tuned-task"
    model_id = "distilbert-base-uncased"  # "dslim/bert-large-NER"   #d"bmdz/bert-large-cased-finetuned-conll03-english"
    logging.set_verbosity_info()
    logger = logging.get_logger("transformers")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, ignore_mismatched_sizes=True, cache_dir=CommandAnalyer.cache_path
    )
    # require model is trained
    instructed_model = None
    pipe = None

    def __init__(self):
        self.foo = None
        api_token = app.config["API_TOKEN"]

    def train(self, train_data):
        fname = f"./data/train/{train_data}"

        if os.path.isfile(fname) == False:
            return False

        with open(fname) as file:
            train_data = file.readlines()
            print(train_data)

        data_collator = DataCollatorForTokenClassification(
            tokenizer=En_Command_Analyzer.tokenizer
        )

        dm = NERDataMaker(train_data)
        self.instructed_model
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_id,
            ignore_mismatched_sizes=True,
            cache_dir=CommandAnalyer.cache_path,
            num_labels=len(dm.unique_entities),
            id2label=dm.id2label,
            label2id=dm.label2id,
        ).to("cuda")

        training_args = TrainingArguments(
            output_dir="./results/distilbert-base-uncased",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=40,
            weight_decay=0.01,
        )

        train_ds = dm.as_hf_dataset(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=train_ds,  # eval on training set! ONLY for DEMO!!
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        print("BEGIN training for model")

        trainer.train()

        print("END training for model")

        trainer.save_model(self.out_dir)

        En_Command_Analyzer.instructed_model = (
            AutoModelForTokenClassification.from_pretrained(
                En_Command_Analyzer.out_dir
            ).to("cuda")
        )

        return True

    def analyze(self, cmd, tzOffset: float = 0):
        if En_Command_Analyzer.instructed_model is None:
            En_Command_Analyzer.instructed_model = (
                AutoModelForTokenClassification.from_pretrained(
                    En_Command_Analyzer.out_dir
                ).to("cuda")
            )

        if En_Command_Analyzer.pipe is None:
            En_Command_Analyzer.pipe = pipeline(
                "ner",
                model=En_Command_Analyzer.instructed_model,
                tokenizer=En_Command_Analyzer.tokenizer,
                aggregation_strategy="none",
                device=0,
            )  # aggregation_strategy = [none, first, simple, average, max ]

        print(f"Command text: {cmd}")

        analyzeCmd = cmd.lower()

        if analyzeCmd[len(analyzeCmd) - 1] == ".":
            analyzeCmd = analyzeCmd[:-1]

        result = En_Command_Analyzer.pipe(analyzeCmd)

        for op in result:
            print(op)

        output_cmd = self.extract_command_data(cmd, result, tzOffset)

        return output_cmd
