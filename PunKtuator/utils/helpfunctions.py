import sys
sys.path.insert(0, '/tilde/vchordia/sentence_boundary/targer/ner-bert')

# from tqdm import tqdm
# import torch
# from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
#                               TensorDataset)
# from pytorch_pretrained_bert import BertTokenizer
# import os
# from healthcheck import HealthCheck
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from modules.data import bert_data_new
from modules.models.bert_models import BERTBiLSTMNCRFJoint
from modules.data.bert_data_new import get_data_loader_for_predict
from modules.analyze_utils.utils import bert_labels2tokens
import re
import pandas as pd
from somajo import Tokenizer
import string
import spacy
from spacy_langdetect import LanguageDetector

nlp = spacy.load("en_core_web_md")
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)



data = bert_data_new.LearnData.create(
    train_df_path="/data/vchordia/sen_boundary/data/spoken/multil/train.csv",
    valid_df_path="/data/vchordia/sen_boundary/data/spoken/multil/val.csv",
    idx2labels_path="../idx2labels.txt",
    idx2cls_path= '../idx2cls.txt',
    is_cls = True,
    clear_cache=True,
    model_name="bert-base-multilingual-cased"
)



model = BERTBiLSTMNCRFJoint.create(
    len(data.train_ds.idx2label), model_name='bert-base-multilingual-cased',
    lstm_dropout=0., crf_dropout=0.3, intent_size=3)


from modules.train.train import NerLearner


num_epochs = 10

learner = NerLearner(
    model, data, "/data/vchordia/sen_boundary/data/spoken/multil/FINAL_MODEL.cpt",
    t_total=num_epochs * len(data.train_dl))

model.get_n_trainable_params()


learner.load_model("/data/vchordia/sen_boundary/data/spoken/multil/FINAL_MODEL.cpt")



def predict(input_text, model = learner):
    # input_txt = ""
    doc = nlp(input_text)
    if 'en' in doc._.language['language']:
        tokenizer = Tokenizer(language="en")
        input_txt = ' '.join(token for token in tokenizer.tokenize_paragraph(input_text) if token not in [',', '.', '?', '!'])
        labels = 'BOS ' * len(tokenizer.tokenize_paragraph(input_txt))

    elif 'de' in doc._.language['language']:
        tokenizer = Tokenizer(split_camel_case=True, token_classes=False, extra_info=False)
        input_txt = ' '.join(token for token in tokenizer.tokenize_paragraph(input_text) if token not in [',', '.', '?', '!'])
        labels = 'BOS ' * len(tokenizer.tokenize_paragraph(input_txt))

    elif 'fr' in doc._.language['language']:
        tokenizer = Tokenizer(language="en")
        input_txt = re.sub(r'[,.?!]', '', input_text).strip()
        labels = 'BOS ' * len(tokenizer.tokenize_paragraph(input_txt))
    else:
        tokenizer = Tokenizer(language="en")
        input_txt = re.sub(r'[,.?!]', '', input_text).strip()
        labels = 'BOS ' * len(tokenizer.tokenize_paragraph(input_txt))

    if not input_txt:
        return input_txt

    ## Assigning random language
    language = 'English'

    X = pd.DataFrame([(input_txt, labels, language)], columns=['Sentences', 'labels', 'language'])
    X.to_csv('/data/vchordia/sen_boundary/X.csv', index=False)
    dl = get_data_loader_for_predict(data, df_path="/data/vchordia/sen_boundary/X.csv")
    preds = learner.predict(dl)
    pred_tokens, pred_labels = bert_labels2tokens(dl, preds[0])
    res_str = final_str(pred_tokens, pred_labels)
    return res_str


def final_str(pred_text, pred_labels):

    res_str = ''
    assert len(pred_text[0]) == len(pred_labels[0])

    for i in range(len(pred_text[0])):
        if pred_labels[0][i] == 'BOS':
            if pred_text[0][i] in ["'s", "'nt", "n't", "'m"]:
                res_str += pred_text[0][i]
            else:
                res_str += ' ' + pred_text[0][i]

        elif pred_labels[0][i] == 'B_O':
            if pred_text[0][i] in ["'s", "'nt", "n't", "'m"]:
                res_str +=  pred_text[0][i]
            else:
                res_str += ' ' + pred_text[0][i]

        elif pred_labels[0][i] == 'O':
            if res_str:
                if res_str[-1] == '.':
                    res_str += ' ' + string.capwords(pred_text[0][i])
                else:
                    if pred_text[0][i] in ["'s", "'nt", "n't", "'m"]:
                        res_str += pred_text[0][i]
                    else:
                        res_str += ' ' + pred_text[0][i]
            else:
                res_str += ' ' + pred_text[0][i]

        elif pred_labels[0][i] == 'COMMA':
            if pred_text[0][i] in ["'s", "'nt", "n't", "'m"]:
                res_str +=  pred_text[0][i] + ','
            else:
                res_str +=  ' ' + pred_text[0][i] + ','


        elif pred_labels[0][i] == 'EXCLAMATION':
            if pred_text[0][i] in ["'s", "'nt", "n't", "'m"]:
                res_str += pred_text[0][i] + '!'
            else:
                res_str += ' ' + pred_text[0][i] + '!'



        elif pred_labels[0][i] == 'PERIOD':
            if pred_text[0][i] in ["'s", "'nt", "n't", "'m"]:
                res_str += pred_text[0][i] + '.'
            else:
                res_str += ' ' + pred_text[0][i] + '.'


        elif pred_labels[0][i] == 'QUESTION':

            if pred_text[0][i] in ["'s", "'nt", "n't", "'m"]:
                res_str += pred_text[0][i] + '?'
            else:
                res_str += ' ' + pred_text[0][i] + '?'


        else:
            if pred_text[0][i] in ["'s", "'nt", "n't", "'m"]:
                res_str += pred_text[0][i]
            else:
                res_str += ' ' + pred_text[0][i]

    return res_str



# def final_str(pred_text, pred_labels):
#
#     res_str = ''
#     assert len(pred_text[0]) == len(pred_labels[0])
#
#     for i in range(len(pred_text[0])):
#         if pred_labels[0][i] == 'BOS':
#             res_str +=  ' ' + pred_text[0][i]
#
#         elif pred_labels[0][i] == 'B_O':
#             res_str += ' ' + pred_text[0][i]
#
#         elif pred_labels[0][i] == 'O':
#             if res_str:
#                 if res_str[-1] == '.':
#                     res_str += ' ' + string.capwords(pred_text[0][i])
#                 else:
#                     res_str += ' ' + pred_text[0][i]
#             else:
#                 res_str += ' ' + pred_text[0][i]
#
#         elif pred_labels[0][i] == 'COMMA':
#             res_str +=  ' ' + pred_text[0][i] + ','
#
#         elif pred_labels[0][i] == 'EXCLAMATION':
#             res_str += ' ' + pred_text[0][i] + '! '
#
#
#         elif pred_labels[0][i] == 'PERIOD':
#             res_str += ' ' + pred_text[0][i] + '.'
#
#         elif pred_labels[0][i] == 'QUESTION':
#
#             res_str += ' ' + pred_text[0][i] + '?'
#
#         else:
#             res_str += ' ' + pred_text[0][i]
#     return res_str
