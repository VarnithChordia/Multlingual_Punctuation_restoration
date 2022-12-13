import os

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import sys
import warnings
from modules.models.bert_models import BERTBiLSTMNCRFJoint
from modules.train.train import NerLearner
from modules.data import bert_data_new
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "../")

data = bert_data_new.LearnData.create(
    train_df_path="/data/",
    valid_df_path="/data/",
    idx2labels_path="../idx2labels.txt",
    idx2cls_path='../idx2cls.txt',
    idx2mode_path='../idx2mode.txt',
    is_cls=True,
    clear_cache=True,
    model_name="bert-base-multilingual-cased"
)

model = BERTBiLSTMNCRFJoint.create(
    len(data.train_ds.idx2label),
    model_name='bert-base-multilingual-cased',
    lstm_dropout=0.,
    crf_dropout=0.3,
    intent_size=3,
    mode_size=2)

num_epochs = 10

learner = NerLearner(
    model,
    data,
    "/data/models/",
    t_total=num_epochs * len(data.train_dl))

model.get_n_trainable_params()

learner.load_model("/data/models/XXXX.cpt")  ## load the model

from modules.data.bert_data_new import get_data_loader_for_predict
from sklearn_crfsuite.metrics import flat_classification_report

from modules.analyze_utils.utils import bert_labels2tokens

languages = ["English", "French", "German"]

mode = ["Written", "Spoken"]

test_df = pd.read_csv('/data/vchordia/sen_boundary/data/final_df/val.csv')

## Running the prediction  for every language and mode and reporting the metrics

for lang in languages:
    for mod in mode:
        tst = test_df[(test_df['language'] == lang) & (test_df['mode'] == mod)]

        tst.to_csv(f"/data/vchordia/sen_boundary/data/test_df_paper/test_{lang}_{mod}.csv", index=False)
        dl = get_data_loader_for_predict(data,
                                         df_path=f"/data/vchordia/sen_boundary/data/test_df_paper/test_{lang}_{mod}.csv")
        preds = learner.predict(dl)

        pred_tokens, pred_labels = bert_labels2tokens(dl, preds[0])
        true_tokens, true_labels = bert_labels2tokens(dl, [x.bert_labels for x in dl.dataset])

        assert pred_tokens == true_tokens
        tokens_report = flat_classification_report(true_labels, pred_labels, labels=['COMMA', 'QUESTION', 'PERIOD'],
                                                   digits=4)

        print(f"The report for {lang}  and {mod} and is {tokens_report}")

