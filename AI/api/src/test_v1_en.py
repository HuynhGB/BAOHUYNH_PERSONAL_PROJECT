import unittest
import sys

# sys.path.insert(1, '/src')
from src import create_app
import json
from types import SimpleNamespace
from datetime import date, timedelta
import base64

url_worknote = "/v1/en/tasks/suggestworknote"
url_workstep = "/v1/en/tasks/suggestworkstep"

class Test_V1_EN(unittest.TestCase):
    headers = {"Content-type": "application/json", "Accept": "application/json"}

    def setUp(self):
        print("setup test.")
        self.app = create_app()
        self.app_ctxt = self.app.app_context()
        self.app_ctxt.push()
        self.client = self.app.test_client()

    def tearDown(self):
        self.app_ctxt.pop()
        self.app = None
        self.app_ctxt = None

    def test_vi_en_conversation(self):
        data = {
            "id": "0a6b24b6-a5d3-433b-95e3-7fa1b2f88257",
            "inputText": "Translate to French: Schools often do not provide students with marketable skills.",
        }

        response = self.client.post(
            "/v1/en/conversation", json=data, headers=self.headers
        )

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert (
            x.outputText
            == " Aucune école ne fournit souvent aux élèves des compétences pouvant être commercialisées."
        )

    def test_vi_en_analyze(self):

        data = {
            "command": "create a meeting with title is Techtalk Q2 2024 with description is We are going to introduce generative AI and meeting time is at 10 am on next monday in 2 hours and forty minutes.",
            "timezoneOffset": 5,
        }

        response = self.client.post("/v1/en/analyze", json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert x.action == "create_event"
        assert x.fields.description == "We are going to introduce generative AI"
        assert x.fields.title == "Techtalk Q2 2024"

    def test_vi_en_answer(self):

        data = {
            "id": "c9dd6ac6-9098-4ffd-b589-4a1a1e027415",
            "question": "When did the founders pack their bags in Norway?",
        }

        response = self.client.post("/v1/en/answer", json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert x.answer == "2006"

    def test_vi_en_recognize(self):

        encoded_string = ""
        with open("audio/a tornado.wav", "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode("ascii")

        data = {"file_data": encoded_string}
        response = self.client.post("/v1/en/recognize", json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert (
            x.output_text
            == "A tornado is a spinning column of very low-pressure air, which sucks the surrounding air inward and upward."
        )

    def test_vi_en_analyze_meeting_on_tomorrow(self):

        data = {
            "command": "create a meeting with title is Techtalk Q2 2024 with description is We are going to introduce generative AI and meeting time is at 10 am on tomorrow in 2 hours and forty minutes.",
            "timezoneOffset": 7,
        }

        response = self.client.post("/v1/en/analyze", json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert x.action == "create_event"
        assert x.fields.description == "We are going to introduce generative AI"
        assert x.fields.title == "Techtalk Q2 2024"

        tomorrow = date.today() + timedelta(days=1)

        assert tomorrow.strftime("%m/%d/%Y") in x.fields.start

    def test_vi_en_analyze_check_in_with_task_name(self):

        taskName = '"Daily scrum meeting"'
        data = {"command": "I would like to begin {0} task.".format(taskName)}

        response = self.client.post("/v1/en/analyze", json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))

        assert x.action == "check_in"
        assert x.fields.title == taskName

    def test_vi_en_analyze_check_in_with_start_task_name(self):

        taskName = '"Daily scrum meeting"'
        data = {"command": "I would like to start {0} task.".format(taskName)}

        response = self.client.post("/v1/en/analyze", json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))

        assert x.action == "check_in"
        assert x.fields.title == taskName

    def test_vi_en_analyze_check_out_by_going_home(self):

        data = {"command": "I am going home now"}

        response = self.client.post("/v1/en/analyze", json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert x.action == "check_out"

    def test_vi_en_analyze_check_out_by_doing_karate(self):

        data = {"command": "I'm going to do karate now", "timezoneOffset": 7}

        response = self.client.post("/v1/en/analyze", json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert x.action == "check_out"

    def test_vi_en_analyze_create_task(self):

        data = {
            "command": "I would like to create task Present Generative AI",
            "timezoneOffset": 7,
        }

        response = self.client.post("/v1/en/analyze", json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert x.action == "create_task"
        assert x.fields.title == "Present Generative AI"

    def test_vi_en_train_with_file_not_existed(self):
        data = {"train_data": "train_data_not_existed.txt"}

        response = self.client.post("/v1/en/train", json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert x.is_trained == False
        
    def test_check_out_with_full_data(self):
        data = {
                "project": "R&D",
                "task": "add new user task to add permission check",
                "switch_type": 1,
                "worknotes": [
                        "Need to update re: office time to show recent completed weeks. Wiki article re: how to register time in a right way.",
                        "Reports changes, from ABC grading.",
                        "Salary review 2024 reports adjustment. On going.",
                        "Done the evaluation, hard to find their contributions.",
                        "Co-employees rating.",
                        "Interpersonal reviews.",
                        "The 'personal'.",
                        "Their other contributions beside the carrying the income.",
                        "Akva cont.",
                        "Tried setup the Review process, automate the review process. Possibility: email templates and mass email sending, 1 template for each reminder with opt out.",
                        "Email templates: automate them; the GG calendar is ready - appointments slot picking.",
                        "Excel vs GG spreadsheet and 365, test display is ok, finding a way for macros.",
                        "Salary review 2024. Automate reminders, templates; do the co-employees reviews. Support Hanh/Binh on excel files.",
                        "Review, with Oystein & Dung. Added notes for improvements with Dung, will finalize and update 3 of us soon."
                    ]
}
        response = self.client.post(url_worknote, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d)) 
        assert len(x.data) == 5
        for i in range(len(x.data)):
            string_of_array = x.data
            assert isinstance(string_of_array[i], str)
        assert len(x.errors) == 0
        assert x.status == 200
        assert x.isSuccess == True


    def test_check_out_with_empty_worknotes(self):
        data = {
                    "project": "Digital Transformation",
                    "task": "Salary review: 2024",
                    "worknotes": []
                }
        response = self.client.post(url_worknote, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d)) 
        assert len(x.data) == 5
        for i in range(len(x.data)):
            string_of_array = x.data
            assert isinstance(string_of_array[i], str)
        assert len(x.errors) == 0
        assert x.status == 200
        assert x.isSuccess == True

    def test_check_out_with_empty_project(self):
        data = {
                    "project": "",
                    "task": "add new user task to add permission check",
                    "worknotes": [
                        "Need to update re: office time to show recent completed weeks. Wiki article re: how to register time in a right way.",
                        "Reports changes, from ABC grading.",
                        "Salary review 2024 reports adjustment. On going.",
                        "Done the evaluation, hard to find their contributions.",
                        "Co-employees rating.",
                        "Interpersonal reviews.",
                        "The 'personal'.",
                        "Their other contributions beside the carrying the income.",
                        "Akva cont.",
                        "Tried setup the Review process, automate the review process. Possibility: email templates and mass email sending, 1 template for each reminder with opt out.",
                        "Email templates: automate them; the GG calendar is ready - appointments slot picking.",
                        "Excel vs GG spreadsheet and 365, test display is ok, finding a way for macros.",
                        "Salary review 2024. Automate reminders, templates; do the co-employees reviews. Support Hanh/Binh on excel files.",
                        "Review, with Oystein & Dung. Added notes for improvements with Dung, will finalize and update 3 of us soon."
                    ]
                }
        response = self.client.post(url_worknote, json=data, headers=self.headers)
        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False


    def test_check_out_with_empty_task(self):
        data = {
                    "project": "R&D",
                    "task": "",
                    "worknotes": [
                        "Need to update re: office time to show recent completed weeks. Wiki article re: how to register time in a right way.",
                        "Reports changes, from ABC grading.",
                        "Salary review 2024 reports adjustment. On going.",
                        "Done the evaluation, hard to find their contributions.",
                        "Co-employees rating.",
                        "Interpersonal reviews.",
                        "The 'personal'.",
                        "Their other contributions beside the carrying the income.",
                        "Akva cont.",
                        "Tried setup the Review process, automate the review process. Possibility: email templates and mass email sending, 1 template for each reminder with opt out.",
                        "Email templates: automate them; the GG calendar is ready - appointments slot picking.",
                        "Excel vs GG spreadsheet and 365, test display is ok, finding a way for macros.",
                        "Salary review 2024. Automate reminders, templates; do the co-employees reviews. Support Hanh/Binh on excel files.",
                        "Review, with Oystein & Dung. Added notes for improvements with Dung, will finalize and update 3 of us soon."
                    ]
}
        response = self.client.post(url_worknote, json=data, headers=self.headers)
        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False


    def test_check_out_with_no_worknotes(self):
        data = {
                    "project": "R&D",
                    "task": "add new user task to add permission check",
                }
        response = self.client.post(url_worknote, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d)) 
        assert len(x.data) == 5
        for i in range(len(x.data)):
            string_of_array = x.data
            assert isinstance(string_of_array[i], str)
        assert len(x.errors) == 0
        assert x.status == 200
        assert x.isSuccess == True


    def test_check_out_with_no_project(self):
        data = {
                    "task": "add new user task to add permission check",
                    "worknotes": [
                        "Need to update re: office time to show recent completed weeks. Wiki article re: how to register time in a right way.",
                        "Reports changes, from ABC grading.",
                        "Salary review 2024 reports adjustment. On going.",
                        "Done the evaluation, hard to find their contributions.",
                        "Co-employees rating.",
                        "Interpersonal reviews.",
                        "The 'personal'.",
                        "Their other contributions beside the carrying the income.",
                        "Akva cont.",
                        "Tried setup the Review process, automate the review process. Possibility: email templates and mass email sending, 1 template for each reminder with opt out.",
                        "Email templates: automate them; the GG calendar is ready - appointments slot picking.",
                        "Excel vs GG spreadsheet and 365, test display is ok, finding a way for macros.",
                        "Salary review 2024. Automate reminders, templates; do the co-employees reviews. Support Hanh/Binh on excel files.",
                        "Review, with Oystein & Dung. Added notes for improvements with Dung, will finalize and update 3 of us soon."
                    ]
                }
        response = self.client.post(url_worknote, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False



    def test_check_out_with_no_task(self):
        data = {
                    "project": "Digital Transformation",
                    "worknotes": [
                        "Need to update re: office time to show recent completed weeks. Wiki article re: how to register time in a right way.",
                        "Reports changes, from ABC grading.",
                        "Salary review 2024 reports adjustment. On going.",
                        "Done the evaluation, hard to find their contributions.",
                        "Co-employees rating.",
                        "Interpersonal reviews.",
                        "The 'personal'.",
                        "Their other contributions beside the carrying the income.",
                        "Akva cont.",
                        "Tried setup the Review process, automate the review process. Possibility: email templates and mass email sending, 1 template for each reminder with opt out.",
                        "Email templates: automate them; the GG calendar is ready - appointments slot picking.",
                        "Excel vs GG spreadsheet and 365, test display is ok, finding a way for macros.",
                        "Salary review 2024. Automate reminders, templates; do the co-employees reviews. Support Hanh/Binh on excel files.",
                        "Review, with Oystein & Dung. Added notes for improvements with Dung, will finalize and update 3 of us soon."
                    ]
                }
        response = self.client.post(url_worknote, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False



    def test_check_out_with_no_project_and_task(self):
        data = {
                    "worknotes": [
                        "Need to update re: office time to show recent completed weeks. Wiki article re: how to register time in a right way.",
                        "Reports changes, from ABC grading.",
                        "Salary review 2024 reports adjustment. On going.",
                        "Done the evaluation, hard to find their contributions.",
                        "Co-employees rating.",
                        "Interpersonal reviews.",
                        "The 'personal'.",
                        "Their other contributions beside the carrying the income.",
                        "Akva cont.",
                        "Tried setup the Review process, automate the review process. Possibility: email templates and mass email sending, 1 template for each reminder with opt out.",
                        "Email templates: automate them; the GG calendar is ready - appointments slot picking.",
                        "Excel vs GG spreadsheet and 365, test display is ok, finding a way for macros.",
                        "Salary review 2024. Automate reminders, templates; do the co-employees reviews. Support Hanh/Binh on excel files.",
                        "Review, with Oystein & Dung. Added notes for improvements with Dung, will finalize and update 3 of us soon."
                    ]
                }
        response = self.client.post(url_worknote, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 2
        assert x.status == 400
        assert x.isSuccess == False

    def test_check_out_with_no_project_and_worknotes(self):
        data = {
                    "task": "Salary review: 2024: Excel/OpenOffice/GG Spreadsheet; Email templates."                 
                }
        response = self.client.post(url_worknote, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False

    def test_check_out_with_no_task_and_worknotes(self):
        data = {
                    "project": "Digital Transformation"
                   
                }
        response = self.client.post(url_worknote, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False
    
    def test_check_out_with_no_data(self):
        data = {}
        response = self.client.post(url_worknote, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 2
        assert x.status == 400
        assert x.isSuccess == False

    def test_check_out_with_empty_project_and_worknotes(self):
        data = {
                    "project": "",
                    "task": "Salary review: 2024: Excel/OpenOffice/GG Spreadsheet; Email templates.",
                    "worknotes": []
                   
                }
        response = self.client.post(url_worknote, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False
    
    def test_check_out_with_empty_project_and_task(self):
        data = {
                    "project": "",
                    "task": "",
                    "worknotes": [
                        "Need to update re: office time to show recent completed weeks. Wiki article re: how to register time in a right way.",
                        "Reports changes, from ABC grading.",
                        "Salary review 2024 reports adjustment. On going.",
                        "Done the evaluation, hard to find their contributions.",
                        "Co-employees rating.",
                        "Interpersonal reviews.",
                        "The 'personal'.",
                        "Their other contributions beside the carrying the income.",
                        "Akva cont.",
                        "Tried setup the Review process, automate the review process. Possibility: email templates and mass email sending, 1 template for each reminder with opt out.",
                        "Email templates: automate them; the GG calendar is ready - appointments slot picking.",
                        "Excel vs GG spreadsheet and 365, test display is ok, finding a way for macros.",
                        "Salary review 2024. Automate reminders, templates; do the co-employees reviews. Support Hanh/Binh on excel files.",
                        "Review, with Oystein & Dung. Added notes for improvements with Dung, will finalize and update 3 of us soon."
                    ]
                   
                }
        response = self.client.post(url_worknote, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 2
        assert x.status == 400
        assert x.isSuccess == False

    def test_check_out_with_empty_task_and_worknotes(self):
        data = {
                    "project": "Digital Transformation",
                    "task": "",
                    "worknotes": []
                   
                }
        response = self.client.post(url_worknote, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False











    def test_check_in_with_full_data(self):
        data = {
                "project": "R&D",
                "task": "add new user task to add permission check",
                "descriptions": [
                        "Need to update re: office time to show recent completed weeks. Wiki article re: how to register time in a right way.",
                        "Reports changes, from ABC grading.",
                        "Salary review 2024 reports adjustment. On going.",
                        "Done the evaluation, hard to find their contributions.",
                        "Co-employees rating.",
                        "Interpersonal reviews.",
                        "The 'personal'.",
                        "Their other contributions beside the carrying the income.",
                        "Akva cont.",
                        "Tried setup the Review process, automate the review process. Possibility: email templates and mass email sending, 1 template for each reminder with opt out.",
                        "Email templates: automate them; the GG calendar is ready - appointments slot picking.",
                        "Excel vs GG spreadsheet and 365, test display is ok, finding a way for macros.",
                        "Salary review 2024. Automate reminders, templates; do the co-employees reviews. Support Hanh/Binh on excel files.",
                        "Review, with Oystein & Dung. Added notes for improvements with Dung, will finalize and update 3 of us soon."
                    ]
}
        response = self.client.post(url_workstep, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d)) 
        assert len(x.data) == 5
        for i in range(len(x.data)):
            string_of_array = x.data
            assert isinstance(string_of_array[i], str)
        assert len(x.errors) == 0
        assert x.status == 200
        assert x.isSuccess == True


    def test_check_in_with_empty_descriptions(self):
        data = {
                    "project": "Digital Transformation",
                    "task": "Salary review: 2024",
                    "descriptions": []
                }
        response = self.client.post(url_workstep, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d)) 
        assert len(x.data) == 5
        for i in range(len(x.data)):
            string_of_array = x.data
            assert isinstance(string_of_array[i], str)
        assert len(x.errors) == 0
        assert x.status == 200
        assert x.isSuccess == True

    def test_check_in_with_empty_project(self):
        data = {
                    "project": "",
                    "task": "add new user task to add permission check",
                    "descriptions": [
                        "Need to update re: office time to show recent completed weeks. Wiki article re: how to register time in a right way.",
                        "Reports changes, from ABC grading.",
                        "Salary review 2024 reports adjustment. On going.",
                        "Done the evaluation, hard to find their contributions.",
                        "Co-employees rating.",
                        "Interpersonal reviews.",
                        "The 'personal'.",
                        "Their other contributions beside the carrying the income.",
                        "Akva cont.",
                        "Tried setup the Review process, automate the review process. Possibility: email templates and mass email sending, 1 template for each reminder with opt out.",
                        "Email templates: automate them; the GG calendar is ready - appointments slot picking.",
                        "Excel vs GG spreadsheet and 365, test display is ok, finding a way for macros.",
                        "Salary review 2024. Automate reminders, templates; do the co-employees reviews. Support Hanh/Binh on excel files.",
                        "Review, with Oystein & Dung. Added notes for improvements with Dung, will finalize and update 3 of us soon."
                    ]
                }
        response = self.client.post(url_workstep, json=data, headers=self.headers)
        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False


    def test_check_in_with_empty_task(self):
        data = {
                    "project": "R&D",
                    "task": "",
                    "descriptions": [
                        "Need to update re: office time to show recent completed weeks. Wiki article re: how to register time in a right way.",
                        "Reports changes, from ABC grading.",
                        "Salary review 2024 reports adjustment. On going.",
                        "Done the evaluation, hard to find their contributions.",
                        "Co-employees rating.",
                        "Interpersonal reviews.",
                        "The 'personal'.",
                        "Their other contributions beside the carrying the income.",
                        "Akva cont.",
                        "Tried setup the Review process, automate the review process. Possibility: email templates and mass email sending, 1 template for each reminder with opt out.",
                        "Email templates: automate them; the GG calendar is ready - appointments slot picking.",
                        "Excel vs GG spreadsheet and 365, test display is ok, finding a way for macros.",
                        "Salary review 2024. Automate reminders, templates; do the co-employees reviews. Support Hanh/Binh on excel files.",
                        "Review, with Oystein & Dung. Added notes for improvements with Dung, will finalize and update 3 of us soon."
                    ]
}
        response = self.client.post(url_workstep, json=data, headers=self.headers)
        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False


    def test_check_in_with_no_descriptions(self):
        data = {
                    "project": "R&D",
                    "task": "add new user task to add permission check",
                }
        response = self.client.post(url_workstep, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d)) 
        assert len(x.data) == 5
        for i in range(len(x.data)):
            string_of_array = x.data
            assert isinstance(string_of_array[i], str)
        assert len(x.errors) == 0
        assert x.status == 200
        assert x.isSuccess == True


    def test_check_in_with_no_project(self):
        data = {
                    "task": "add new user task to add permission check",
                    "descriptions": [
                        "Need to update re: office time to show recent completed weeks. Wiki article re: how to register time in a right way.",
                        "Reports changes, from ABC grading.",
                        "Salary review 2024 reports adjustment. On going.",
                        "Done the evaluation, hard to find their contributions.",
                        "Co-employees rating.",
                        "Interpersonal reviews.",
                        "The 'personal'.",
                        "Their other contributions beside the carrying the income.",
                        "Akva cont.",
                        "Tried setup the Review process, automate the review process. Possibility: email templates and mass email sending, 1 template for each reminder with opt out.",
                        "Email templates: automate them; the GG calendar is ready - appointments slot picking.",
                        "Excel vs GG spreadsheet and 365, test display is ok, finding a way for macros.",
                        "Salary review 2024. Automate reminders, templates; do the co-employees reviews. Support Hanh/Binh on excel files.",
                        "Review, with Oystein & Dung. Added notes for improvements with Dung, will finalize and update 3 of us soon."
                    ]
                }
        response = self.client.post(url_workstep, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False



    def test_check_in_with_no_task(self):
        data = {
                    "project": "Digital Transformation",
                    "descriptions": [
                        "Need to update re: office time to show recent completed weeks. Wiki article re: how to register time in a right way.",
                        "Reports changes, from ABC grading.",
                        "Salary review 2024 reports adjustment. On going.",
                        "Done the evaluation, hard to find their contributions.",
                        "Co-employees rating.",
                        "Interpersonal reviews.",
                        "The 'personal'.",
                        "Their other contributions beside the carrying the income.",
                        "Akva cont.",
                        "Tried setup the Review process, automate the review process. Possibility: email templates and mass email sending, 1 template for each reminder with opt out.",
                        "Email templates: automate them; the GG calendar is ready - appointments slot picking.",
                        "Excel vs GG spreadsheet and 365, test display is ok, finding a way for macros.",
                        "Salary review 2024. Automate reminders, templates; do the co-employees reviews. Support Hanh/Binh on excel files.",
                        "Review, with Oystein & Dung. Added notes for improvements with Dung, will finalize and update 3 of us soon."
                    ]
                }
        response = self.client.post(url_workstep, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False



    def test_check_in_with_no_project_and_task(self):
        data = {
                    "descriptions": [
                        "Need to update re: office time to show recent completed weeks. Wiki article re: how to register time in a right way.",
                        "Reports changes, from ABC grading.",
                        "Salary review 2024 reports adjustment. On going.",
                        "Done the evaluation, hard to find their contributions.",
                        "Co-employees rating.",
                        "Interpersonal reviews.",
                        "The 'personal'.",
                        "Their other contributions beside the carrying the income.",
                        "Akva cont.",
                        "Tried setup the Review process, automate the review process. Possibility: email templates and mass email sending, 1 template for each reminder with opt out.",
                        "Email templates: automate them; the GG calendar is ready - appointments slot picking.",
                        "Excel vs GG spreadsheet and 365, test display is ok, finding a way for macros.",
                        "Salary review 2024. Automate reminders, templates; do the co-employees reviews. Support Hanh/Binh on excel files.",
                        "Review, with Oystein & Dung. Added notes for improvements with Dung, will finalize and update 3 of us soon."
                    ]
                }
        response = self.client.post(url_workstep, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 2
        assert x.status == 400
        assert x.isSuccess == False

    def test_check_in_with_no_project_and_descriptions(self):
        data = {
                    "task": "Salary review: 2024: Excel/OpenOffice/GG Spreadsheet; Email templates."                 
                }
        response = self.client.post(url_workstep, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False

    def test_check_in_with_no_task_and_descriptions(self):
        data = {
                    "project": "Digital Transformation"
                   
                }
        response = self.client.post(url_workstep, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False
    
    def test_check_in_with_no_data(self):
        data = {}
        response = self.client.post(url_workstep, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 2
        assert x.status == 400
        assert x.isSuccess == False

    def test_check_inwith_empty_project_and_descriptions(self):
        data = {
                    "project": "",
                    "task": "Salary review: 2024: Excel/OpenOffice/GG Spreadsheet; Email templates.",
                    "descriptions": []
                   
                }
        response = self.client.post(url_workstep, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False
    
    def test_check_in_with_empty_project_and_task(self):
        data = {
                    "project": "",
                    "task": "",
                    "descriptions": [
                        "Need to update re: office time to show recent completed weeks. Wiki article re: how to register time in a right way.",
                        "Reports changes, from ABC grading.",
                        "Salary review 2024 reports adjustment. On going.",
                        "Done the evaluation, hard to find their contributions.",
                        "Co-employees rating.",
                        "Interpersonal reviews.",
                        "The 'personal'.",
                        "Their other contributions beside the carrying the income.",
                        "Akva cont.",
                        "Tried setup the Review process, automate the review process. Possibility: email templates and mass email sending, 1 template for each reminder with opt out.",
                        "Email templates: automate them; the GG calendar is ready - appointments slot picking.",
                        "Excel vs GG spreadsheet and 365, test display is ok, finding a way for macros.",
                        "Salary review 2024. Automate reminders, templates; do the co-employees reviews. Support Hanh/Binh on excel files.",
                        "Review, with Oystein & Dung. Added notes for improvements with Dung, will finalize and update 3 of us soon."
                    ]
                   
                }
        response = self.client.post(url_workstep, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 2
        assert x.status == 400
        assert x.isSuccess == False

    def test_check_in_with_empty_task_and_descriptions(self):
        data = {
                    "project": "Digital Transformation",
                    "task": "",
                    "descriptions": []
                   
                }
        response = self.client.post(url_workstep, json=data, headers=self.headers)

        assert response.status_code == 200
        x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
        assert len(x.errors) == 1
        assert x.status == 400
        assert x.isSuccess == False