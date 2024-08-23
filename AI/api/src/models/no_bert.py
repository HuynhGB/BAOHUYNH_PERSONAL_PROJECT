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
from src.apis.datetime_utils import DatetimeUtils
from datetime import datetime
from dateutil import tz
from pytz import utc


class CommandAnalyer:
    action_unknown = "unknown"
    action_create_task = "create_task"
    action_create_event = "create_event"
    action_check_in = "check_in"
    action_check_out = "check_out"

    cache_path = "./cache/"

    def get_action(self, result=[]):
        action = None

        if len(result) > 1:

            if result[0]["entity"] == "B-create":
                lastPropToken = self.get_last_property_token(result[0], result)
                nextEntityTokenIdx = result.index(lastPropToken) + 1
                if nextEntityTokenIdx < len(result):
                    if result[nextEntityTokenIdx]["entity"] == "B-task":
                        return self.action_create_task
                    elif result[nextEntityTokenIdx]["entity"] == "B-event":
                        return self.action_create_event

        return action

    def get_last_property_token(self, entity_token, result):

        if entity_token is not None and len(result) > 1:
            last_property_token = entity_token
            property_name = entity_token["entity"]
            property_name = property_name.replace("B", "I", 1)
            for idx, x in enumerate(result):
                if result.index(x) > result.index(last_property_token):
                    if (
                        result.index(x) == result.index(last_property_token) + 1
                        and property_name == x["entity"]
                    ):
                        last_property_token = x
                    else:
                        return last_property_token

            return last_property_token

        return entity_token

    def get_content_of_token(self, entityToken, cmd, result=[]):
        content = ""

        if entityToken in result:
            lastPropToken = self.get_last_property_token(entityToken, result)
            nextEntityTokenIdx = result.index(lastPropToken) + 1
            startIdx = lastPropToken["end"]
            if nextEntityTokenIdx < len(result):
                endIdx = result[nextEntityTokenIdx]["start"]
                content = cmd[startIdx:endIdx]
            else:
                content = cmd[startIdx:]

        return content

    def get_field_content(self, field, cmd, result=[]):
        entityToken = Enumerable(result).first_or_default(
            lambda i: i["entity"] == "B-" + field
        )

        if entityToken is not None:
            return self.get_content_of_token(entityToken, cmd, result).strip()
        else:
            return ""

    def extract_event_data(self, cmd, result, tzOffset=0):
        eventTime = self.get_field_content("event_time", cmd, result).lower()

        if len(eventTime) > 0 and eventTime[len(eventTime) - 1] == ".":
            eventTime = eventTime[:-1]

        timeArray = eventTime.split(" ")
        strStart = strEnd = None
        if len(timeArray) > 0:

            atTime = DatetimeUtils.extract_time(timeArray)
            fromDate = DatetimeUtils.extract_date(timeArray)
            duration = DatetimeUtils.extract_duration(timeArray)

            if atTime is not None and fromDate is not None and duration is not None:
                startDateTime = datetime(
                    day=fromDate.day,
                    month=fromDate.month,
                    year=fromDate.year,
                    hour=atTime.hour,
                    minute=atTime.minute,
                    tzinfo=tz.tzoffset(None, tzOffset * 60 * 60),
                )

                endDateTime = startDateTime + duration

                strStart = startDateTime.astimezone(utc).strftime("%m/%d/%Y, %H:%M:%S")
                strEnd = endDateTime.astimezone(utc).strftime("%m/%d/%Y, %H:%M:%S")

        return {
            "action": self.action_create_event,
            "fields": {
                "title": self.get_field_content("title", cmd, result),
                "description": self.get_field_content("description", cmd, result),
                "guests": self.get_field_content("guests", cmd, result),
                "start": strStart,
                "end": strEnd,
            },
        }

    def extract_command_data(self, cmd, result, tzOffset=0):

        action = self.get_action(result)

        if action == self.action_create_event:
            return self.extract_event_data(cmd, result, tzOffset)

        if len(result) > 1:
            if result[0]["entity"] == "B-create" and Enumerable(result).any(
                lambda i: i["entity"] == "B-task"
            ):
                # checkToken = result[0]
                checkToken = self.get_last_property_token(result[0], result)

                firstToken = Enumerable(result).first(lambda i: i["entity"] == "B-task")
                lastToken = Enumerable(result).last(lambda i: i["entity"] == "B-task")
                if firstToken["index"] == checkToken["index"] + 1:
                    # print('Create task: ' + cmd[firstToken['end'] + 1:])
                    # title = cmd[firstToken['end'] + 1:]
                    taskkPropToken = self.get_last_property_token(firstToken, result)
                    title = cmd[taskkPropToken["end"] + 1 :]
                    return {
                        "action": self.action_create_task,
                        "fields": {"title": StringUtils.title_line(title)},
                    }

                else:
                    # print('Create task: ' + cmd[checkToken['end'] + 1: lastToken['start']])
                    title = cmd[checkToken["end"] + 1 : lastToken["start"]]
                    return {
                        "action": self.action_create_task,
                        "fields": {"title": StringUtils.title_line(title)},
                    }

            elif result[0]["entity"] == "B-check":
                checkToken = self.get_last_property_token(result[0], result)

                checkIndex = result.index(checkToken)
                ioToken = result[checkIndex + 1]  # in or out token
                ioPopToken = self.get_last_property_token(ioToken, result)

                isContainTaskToken = Enumerable(result).any(
                    lambda i: i["entity"] == "B-task"
                )

                if ioToken["entity"] == "B-in" or ioToken["entity"] == "B-out":
                    action = self.action_check_in
                    if ioToken["entity"] == "B-out":
                        action = self.action_check_out

                    if isContainTaskToken:
                        firstToken = Enumerable(result).first(
                            lambda i: i["entity"] == "B-task"
                        )
                        lastToken = Enumerable(result).last(
                            lambda i: i["entity"] == "B-task"
                        )
                        if result.index(firstToken) == result.index(ioPopToken) + 1:
                            # print('Check in task: ' + cmd[firstToken['end'] + 1:])
                            taskkPropToken = self.get_last_property_token(
                                firstToken, result
                            )
                            title = cmd[taskkPropToken["end"] + 1 :]
                            return {
                                "action": action,
                                "fields": {"title": StringUtils.title_line(title)},
                            }
                        else:
                            # print('Check in task: ' + cmd[outToken['end'] + 1: lastToken['start']])
                            title = cmd[ioToken["end"] + 1 : lastToken["start"]]
                            return {
                                "action": action,
                                "fields": {"title": StringUtils.title_line(title)},
                            }
                    else:
                        # print('Check in task: ' + cmd[outToken['end'] + 1:])
                        title = cmd[ioToken["end"] + 1 :]
                        return {
                            "action": action,
                            "fields": {"title": StringUtils.title_line(title)},
                        }
                else:
                    print("x")

            elif result[0]["entity"] == "B-check_in":
                check_in_Token = self.get_last_property_token(result[0], result)

                isContainTaskToken = Enumerable(result).any(
                    lambda i: i["entity"] == "B-task"
                )

                if isContainTaskToken:
                    firstToken = Enumerable(result).first(
                        lambda i: i["entity"] == "B-task"
                    )
                    lastToken = Enumerable(result).last(
                        lambda i: i["entity"] == "B-task"
                    )
                    if firstToken["index"] == check_in_Token["index"] + 1:
                        # print('Check in task: ' + cmd[firstToken['end'] + 1:])
                        # title = cmd[firstToken['end'] + 1:]
                        taskkPropToken = self.get_last_property_token(
                            firstToken, result
                        )
                        title = cmd[taskkPropToken["end"] + 1 :]
                        return {
                            "action": self.action_check_in,
                            "fields": {"title": StringUtils.title_line(title)},
                        }
                    else:
                        # print('Check in task: ' + cmd[check_out_Token['end'] + 1: lastToken['start']])
                        title = cmd[check_in_Token["end"] + 1 : lastToken["start"]]
                        return {
                            "action": self.action_check_in,
                            "fields": {"title": StringUtils.title_line(title)},
                        }
                else:
                    # print('Check in task: ' + cmd[check_out_Token['end'] + 1:])
                    title = cmd[check_in_Token["end"] + 1 :]
                    return {
                        "action": self.action_check_in,
                        "fields": {"title": StringUtils.title_line(title)},
                    }

            elif result[0]["entity"] == "B-check" and result[1]["entity"] == "B-out":
                checkToken = result[0]
                outToken = result[1]
                isContainTaskToken = Enumerable(result).any(
                    lambda i: i["entity"] == "B-task"
                )

                if outToken["start"] == checkToken["end"] + 1:
                    if isContainTaskToken:
                        firstToken = Enumerable(result).first(
                            lambda i: i["entity"] == "B-task"
                        )
                        lastToken = Enumerable(result).last(
                            lambda i: i["entity"] == "B-task"
                        )
                        if firstToken["index"] == outToken["index"] + 1:
                            print("Check out task: " + cmd[firstToken["end"] + 1 :])
                            title = cmd[firstToken["end"] + 1 :]
                            return {
                                "action": self.action_check_out,
                                "fields": {"title": StringUtils.title_line(title)},
                            }

                        else:
                            print(
                                "Check out task: "
                                + cmd[outToken["end"] + 1 : lastToken["start"]]
                            )
                            title = cmd[outToken["end"] + 1 : lastToken["start"]]
                            return {
                                "action": self.action_check_out,
                                "fields": {"title": StringUtils.title_line(title)},
                            }
                    else:
                        print("Check out task: " + cmd[outToken["end"] + 1 :])
                        title = cmd[outToken["end"] + 1 :]
                        return {
                            "action": self.action_check_out,
                            "fields": {"title": StringUtils.title_line(title)},
                        }

            elif result[0]["entity"] == "B-check_out":
                print("Check out current task if it is already checked in.")
                return {"action": self.action_check_out, "fields": []}
            else:
                print("Unable to analyze command: " + cmd)
                return {"action": self.action_unknown, "fields": []}
        else:
            print("Unable to analyze command: " + cmd)
            return {"action": self.action_unknown, "fields": []}


class No_Command_Analyzer(CommandAnalyer):
    out_dir = "./models/nb-bert-base-ner-fine-tuned-task"
    logging.set_verbosity_info()
    logger = logging.get_logger("transformers")
    model_id = "NbAiLab/nb-bert-base-ner"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=CommandAnalyer.cache_path
    )
    # require model is trained
    instructed_model = None
    pipe = None

    def __init__(self):
        self.foo = None

    def train(self, train_data):

        with open("./data/train/ner-no.txt") as file:
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
            output_dir="./results/nb-bert-base-ner",
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
        if No_Command_Analyzer.instructed_model is None:
            No_Command_Analyzer.instructed_model = (
                AutoModelForTokenClassification.from_pretrained(self.out_dir).to("cuda")
            )

        if No_Command_Analyzer.pipe is None:
            No_Command_Analyzer.pipe = pipeline(
                "ner",
                model=No_Command_Analyzer.instructed_model,
                tokenizer=self.tokenizer,
                aggregation_strategy="none",
                device=0,
            )  # aggregation_strategy = [none, first, simple, average, max ]

        print(f"Command text: {cmd}")
        result = No_Command_Analyzer.pipe(cmd)
        output_cmd = super().extract_command_data(cmd, result)

        return output_cmd
