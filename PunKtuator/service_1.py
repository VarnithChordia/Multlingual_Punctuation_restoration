import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.insert(0,'../')
from tqdm import tqdm
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_pretrained_bert import BertTokenizer
from modules.data import bert_data_new
from modules.models.bert_models import BERTBiLSTMNCRFJoint
from modules.data.bert_data_new import get_data_loader_for_predict

from healthcheck import HealthCheck
from modules.analyze_utils.utils import bert_labels2tokens

import re
import pandas as pd
import string
from flask import Flask, request, abort, jsonify, render_template
from flask_cors import CORS

from utils.helpfunctions import predict, learner



# text = ""

# predict(text)


# predict('Where are you going')

app = Flask(__name__, static_folder="./PunKtuator-frontend/build/static", template_folder="./PunKtuator-frontend/build")
CORS(app)

health = HealthCheck()
app.add_url_rule("/healthcheck", "healthcheck", view_func=lambda: health.run())


@app.route('/punctuate', methods=['POST'])
def punctuate():
    req = request.get_json()
    input_text = req['input_text']

    output_text = predict(input_text, model = learner)

    data_json = {
        "text": output_text
    }

    return jsonify(data_json)

# @app.route('/annotate', methods=['POST'])
# def annotate():@app.route('/annotate', methods=['POST'])
# def annotate():



# app = Flask(__name__, static_folder="../frontend/build/static", template_folder="../frontend/build")  # noqa


#react frontend

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def frontend(path):
    return render_template("index.html")


if __name__ == '__main__':
    app.run("0.0.0.0", "5000")

