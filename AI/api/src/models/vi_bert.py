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
from src.stringutils import StringUtils
from src.models.no_bert import CommandAnalyer


class Vi_Command_Analyzer(CommandAnalyer):
    out_dir = "./models/NlpHUST/ner-vietnamese-electra-base"
    logging.set_verbosity_info()
    logger = logging.get_logger("transformers")
    model_id = "NlpHUST/ner-vietnamese-electra-base"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=CommandAnalyer.cache_path
    )
    # require model is trained
    instructed_model = None
    pipe = None

    def __init__(self):
        self.foo = None

    def train(self, train_data):

        with open("./data/train/ner-vi.txt", encoding="utf8") as file:
            train_data = file.readlines()
            print(train_data)

        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        dm = NERDataMaker(train_data)

        model = AutoModelForTokenClassification.from_pretrained(
            self.model_id,
            cache_dir=self.cache_path,
            ignore_mismatched_sizes=True,
            num_labels=len(dm.unique_entities),
            id2label=dm.id2label,
            label2id=dm.label2id,
        ).to("cuda")

        training_args = TrainingArguments(
            output_dir="./results/NlpHUST/ner-vietnamese-electra-base",
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
        print("BEGIn training for model")

        trainer.train()

        print("END training for model")

        trainer.save_model(self.out_dir)

        self.instructed_model = AutoModelForTokenClassification.from_pretrained(
            self.out_dir
        ).to("cuda")

        return True

    def analyze(self, cmd):
        if Vi_Command_Analyzer.instructed_model is None:
            Vi_Command_Analyzer.instructed_model = (
                AutoModelForTokenClassification.from_pretrained(self.out_dir).to("cuda")
            )

        if Vi_Command_Analyzer.pipe is None:
            Vi_Command_Analyzer.pipe = pipeline(
                "ner",
                model=Vi_Command_Analyzer.instructed_model,
                tokenizer=self.tokenizer,
                aggregation_strategy="none",
                device=0,
            )  # aggregation_strategy = [none, first, simple, average, max ]

        print(f"Command text: {cmd}")
        result = Vi_Command_Analyzer.pipe(cmd)
        output_cmd = super().extract_command_data(cmd, result)

        return output_cmd
