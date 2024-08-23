from flask import jsonify, request
from flask_cors import cross_origin
from src.models.flanT5 import ask_model
from src.models.roberta import answer
from src.models.wav21 import openai_recognize
from src.models.en_bert import En_Command_Analyzer
from flask import jsonify, request
from flask_cors import cross_origin
from flask import Blueprint
from src.apis.datetime_utils import DatetimeUtils
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.llms import Ollama, OpenAI
from langchain_openai import OpenAIEmbeddings
import json
import pandas as pd
import numpy as np
import os
import uuid
import re
import sys
from create_table import PDF
from dotenv import load_dotenv 
load_dotenv()


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if sys.platform.startswith("linux"):  # could be "linux", "linux2", "linux3", ...
    # linux
    from datasets import load_dataset
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

v1_en = Blueprint("v1_en", __name__)

@v1_en.route("/v1/en/conversation", methods=["POST"])
@cross_origin()
def v1_en_conversation():
    req_data = request.get_json()
    input_text = req_data["inputText"]
    id = ""
    if "id" in req_data:
        id = req_data["id"]
    print(f'INFO: Request:"{input_text}" with id:{id}')
    output_text = ask_model(input_text)
    responseText = output_text.replace("<pad>", "").replace("</s>", "")
    print(f'INFO: Response:"{responseText}"')
    return jsonify(id=id, outputText=responseText)


@v1_en.route("/v1/en/answer", methods=["POST"])
@cross_origin()
def v1_en_answer():
    req_data = request.get_json()

    question = req_data["question"]
    id = ""
    if "id" in req_data:
        id = req_data["id"]
    output_text = answer(question, "companyInfo")
    responseText = output_text.replace("<pad>", "").replace("</s>", "")
    return jsonify(id=id, answer=responseText)

def process_string(result, split_result, array_of_result):
    trans_table = {ord('\"') : None, ord('.') : None}
    if ( '*' in result):
        for i in range (len(split_result)):
            line = split_result[i]
            if len(line) > 0 and line[0] in "*":
                line = line.replace(line[0:2],'').translate(trans_table)
                array_of_result.append(line)
    else:
        for i in range (len(split_result)):
            line = split_result[i]
            if len(line) > 0 and line[0] in "12345":
                line = line.replace(line[0:3],'').translate(trans_table)
                print(line)
                array_of_result.append(line)

@v1_en.route("/v1/en/tasks/suggestworknote", methods=["POST"])
@cross_origin()
def v1_en_checkout():
    req_data = request.get_json()
    timesheet_file = f"data/checkout_data/{uuid.uuid4().hex}.pdf"
    timesheet_folder = "data/checkout_data"


    pdf = PDF('L', 'mm', 'Letter')
    pdf.add_page()
    pdf.set_font('helvetica','',8)


    project = ""
    task = ""
    worknotes = [] 
    array_result = []
    switch_type = 1

    if "switch_type" in req_data:
        if (req_data["switch_type"] == "") or (req_data["switch_type"] == 0):
            switch_type = 1
        else:
            switch_type = req_data["switch_type"]
    
    
    if "project" in req_data:
        project = req_data['project']
    if "task" in req_data:
        task = req_data['task']
    if "worknotes" in req_data:
        worknotes = req_data['worknotes']
    error_array = []
    if (project == ""):
        error_array.append("Project is required")
    if (task == ""):
        error_array.append("Task is required")
    if len(error_array) > 0:
        return jsonify(status = 400, isSuccess = False, errors = error_array)
    
    pdf.cell(40,10,"project: " + project, ln = True)
    pdf.cell(40,10,"task: " + task, ln = True)
    if(len(worknotes) > 0):
        pdf.cell(40,10,"worknotes: ", ln= True)
        for i in range (len(worknotes)):
            pdf.cell(40,10,"." + worknotes[i], ln= True)
    
    
    if os.path.exists(timesheet_folder):
        pdf.output(timesheet_file)
    else:
        os.makedirs(timesheet_folder)
        pdf.output(timesheet_file)    

    loaders = PyPDFLoader(timesheet_file)
    documents = loaders.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    if (switch_type == 1):
        try:
            Ollama_embeddings = OllamaEmbeddings(model="llama3", base_url = "http://ovn-srv0209:11434")
            persist_directory="Chroma_for_llama3_vectorstore_in_checkout"
            vectorstore = Chroma.from_documents(documents=texts, embedding=Ollama_embeddings, persist_directory=persist_directory)

            llm = Ollama(model="llama3", base_url = "http://ovn-srv0209:11434")

            question = f"Based on given context, try to generate five relevant worknotes for the task{task} of project{project} and return as a list of strings."
            qachain=RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
            res = qachain.invoke({"query": question})
            print(res['result'])
            res1 = res['result'].split('\n')


            process_string(res['result'], res1, array_result)

            print(array_result)
            print(len(array_result))

            ids = vectorstore.get(where = {'source': timesheet_file})['ids']
            print("before: ", ids)
            vectorstore.delete(ids)
            ids = vectorstore.get(where = {'source': timesheet_file})['ids']
            print("after: ", ids)

        except:
            if os.path.exists(timesheet_file):
                os.remove(timesheet_file)
            return jsonify(status = 400, isSuccess = False, errors = [{"Input": "None","Message": "Check your port connection to Ollama!"}])

    elif (switch_type == 2):
        try:
            OpenAI_embeddings = OpenAIEmbeddings()
            persist_directory="Chroma_for_openai_vectorstore_in_checkout"
            vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAI_embeddings, persist_directory=persist_directory)
            llm = OpenAI()
            question = f"Based on given context, try to generate five relevant worknotes for the task{task} of project{project} and return as a list of strings."
            qachain=RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
            res = qachain.invoke({"query": question})
            print(res['result'])
            res1 = res['result'].split('\n')

            process_string(res['result'], res1, array_result)

            print(array_result)
            print(len(array_result))

            ids = vectorstore.get(where = {'source': timesheet_file})['ids']
            print("before: ", ids)
            vectorstore.delete(ids)
            ids = vectorstore.get(where = {'source': timesheet_file})['ids']
            print("after: ", ids)
 
        except:
            if os.path.exists(timesheet_file):
                os.remove(timesheet_file)
            return jsonify(status = 400, isSuccess = False, errors = [{"Input": "None","Message": "Check your balance in your api key!"}])
    if os.path.exists(timesheet_file):
        os.remove(timesheet_file)

    return jsonify(data = array_result,status = 200, isSuccess = True, errors = [])
 

@v1_en.route("/v1/en/tasks/suggestworkstep", methods=["POST"])
@cross_origin()
def v1_en_answercheckin():
    req_data = request.get_json()
    timesheet_file = f"data/checkin_data/{uuid.uuid4().hex}.pdf"
    timesheet_folder = "data/checkin_data"


    pdf = PDF('L', 'mm', 'Letter')
    pdf.add_page()
    pdf.set_font('helvetica','',8)


    project = ""
    task = ""
    descriptions = [] 
    array_result = []
    switch_type = 1

    if "switch_type" in req_data:
        if (req_data["switch_type"] == "") or (req_data["switch_type"] == 0):
            switch_type = 1
        else:
            switch_type = req_data["switch_type"]
    
    
    if "project" in req_data:
        project = req_data['project']
    if "task" in req_data:
        task = req_data['task']
    if "descriptions" in req_data:
       descriptions = req_data['descriptions']
    error_array = []
    if (project == ""):
        error_array.append("Project is required")
    if (task == ""):
        error_array.append("Task is required")
    if len(error_array) > 0:
        return jsonify(status = 400, isSuccess = False, errors = error_array)
    
    pdf.cell(40,10,"project: " + project, ln = True)
    pdf.cell(40,10,"task: " + task, ln = True)
    if(len(descriptions) > 0):
        pdf.cell(40,10,"descriptions: ", ln= True)
        for i in range (len(descriptions)):
            pdf.cell(40,10,"." + descriptions[i], ln= True)
    
    
    if os.path.exists(timesheet_folder):
        pdf.output(timesheet_file)
    else:
        os.makedirs(timesheet_folder)
        pdf.output(timesheet_file)    

    loaders = PyPDFLoader(timesheet_file)
    documents = loaders.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    if (switch_type == 1):
        try:
            Ollama_embeddings = OllamaEmbeddings(model="llama3", base_url = "http://ovn-srv0209:11434")
            persist_directory="Chroma_for_llama3_vectorstore_in_checkin"
            vectorstore = Chroma.from_documents(documents=texts, embedding=Ollama_embeddings, persist_directory=persist_directory)

            llm = Ollama(model="llama3", base_url = "http://ovn-srv0209:11434")

            question = f"Based on given context, try to generate five next steps for the task{task} of project{project} and return as a list of strings."
            qachain=RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
            res = qachain.invoke({"query": question})
            print(res['result'])
            res1 = res['result'].split('\n')


            process_string(res['result'], res1, array_result)

            print(array_result)
            print(len(array_result))

            ids = vectorstore.get(where = {'source': timesheet_file})['ids']
            print("before: ", ids)
            vectorstore.delete(ids)
            ids = vectorstore.get(where = {'source': timesheet_file})['ids']
            print("after: ", ids)

        except:
            if os.path.exists(timesheet_file):
                os.remove(timesheet_file)
            return jsonify(status = 400, isSuccess = False, errors = [{"Input": "None","Message": "Check your port connection to Ollama!"}])

    elif (switch_type == 2):
        try:
            OpenAI_embeddings = OpenAIEmbeddings()
            persist_directory="Chroma_for_openai_vectorstore_in_checkin"
            vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAI_embeddings, persist_directory=persist_directory)
            llm = OpenAI()
            question = f"Based on given context, try to generate five next steps for the task{task} of project{project} and return as a list of strings."
            qachain=RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
            res = qachain.invoke({"query": question})
            print(res['result'])
            res1 = res['result'].split('\n')


            process_string(res['result'], res1, array_result)

            print(array_result)
            print(len(array_result))

            ids = vectorstore.get(where = {'source': timesheet_file})['ids']
            print("before: ", ids)
            vectorstore.delete(ids)
            ids = vectorstore.get(where = {'source': timesheet_file})['ids']
            print("after: ", ids)
 
        except:
            if os.path.exists(timesheet_file):
                os.remove(timesheet_file)
            return jsonify(status = 400, isSuccess = False, errors = [{"Input": "None","Message": "Check your balance in your api key!"}])
    if os.path.exists(timesheet_file):
        os.remove(timesheet_file)

    return jsonify(data = array_result,status = 200, isSuccess = True, errors = [])
    

@v1_en.route("/v1/en/recognize", methods=["POST"])
@cross_origin()
def v1_en_recognize():
    req_data = request.get_json()

    question = req_data["file_data"]

    output_text = openai_recognize(question)
    return jsonify(output_text=output_text.strip())


@v1_en.route("/v1/en/analyze", methods=["POST"])
@cross_origin()
def v1_en_analyze():
    req_data = request.get_json()

    cmd = req_data["command"]
    tzOffset = 0

    if "timezoneOffset" in req_data:
        timezoneOffset = req_data["timezoneOffset"]
        tzOffset = float(timezoneOffset)

    ca = En_Command_Analyzer()
    output_cmd = ca.analyze(cmd, tzOffset=tzOffset)
    print(f"Analyzed command: {output_cmd}")

    return jsonify(output_cmd)


@v1_en.route("/v1/en/train", methods=["POST"])
@cross_origin()
def v1_en_train():
    req_data = request.get_json()

    train_data = req_data["train_data"]
    ca = En_Command_Analyzer()
    is_trained = ca.train(train_data)
    return jsonify(is_trained=is_trained)


trueValues = [
    "yes",
    "submit",
    "of course",
    "yeah",
    "ok",
    "well",
    "let do it",
    "confirm",
    "yes, of course",
    "okay",
]
falseValues = ["no", "cancel", "hold on", "give me a second"]


@v1_en.route("/v1/en/convert", methods=["POST"])
@cross_origin()
def v1_en_convert():
    req_data = request.get_json()

    if "text" not in req_data or "toType" not in req_data:
        return jsonify(value=None)

    text = req_data["text"]
    toType = req_data["toType"]

    text = DatetimeUtils.enhance_text(text, toType)

    if not isinstance(text, str) or not isinstance(toType, str):
        return jsonify(value=None)

    if toType.lower() == "time":
        txtArray = text.split(" ") if isinstance(text, str) else []
        if "at" not in txtArray:
            txtArray.insert(0, "at")

        result = DatetimeUtils.extract_time(txtArray)

        if result is None:
            return jsonify(value=None)

        return jsonify(value=str(result))

    elif toType.lower() == "date":
        try:
            result = datetime.strptime(text, "%b %d, %Y").date()
            return jsonify(value=str(result))
        except:
            print(f"input datetime string: {text}")

        txtArray = text.split(" ") if isinstance(text, str) else []

        if "on" not in txtArray:
            txtArray.insert(0, "on")

        result = DatetimeUtils.extract_date(txtArray)

        if result is None:
            return jsonify(value=None)

        return jsonify(value=str(result))

    elif toType.lower() == "duration":
        txtArray = text.split(" ") if isinstance(text, str) else []

        if "in" not in txtArray:
            txtArray.insert(0, "in")

        result = DatetimeUtils.extract_duration(txtArray)

        if result is None:
            return jsonify(value=None)

        return jsonify(value=str(result))

    elif toType.lower() == "boolean":
        text = text.strip().lower()

        if text in trueValues:
            return jsonify(value="true")

        if text in falseValues:
            return jsonify(value="false")

    return jsonify(value=None)
