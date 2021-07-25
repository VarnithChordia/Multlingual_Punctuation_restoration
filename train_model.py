import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import warnings
from modules.models.bert_models import BERTBiLSTMNCRFJoint
from modules.train.train import NerLearner
from modules.data import bert_data_new

warnings.filterwarnings("ignore")
sys.path.insert(0, "../")


data = bert_data_new.LearnData.create(
    train_df_path="/data/", ## Location of the train dataset
    valid_df_path="/data/", ## Location of the val dataset
    idx2labels_path="../idx2labels.txt", ## Location to store punctuation labels
    idx2cls_path= '../idx2cls.txt', ## Location to store language labels
    idx2mode_path='../idx2mode.txt', ## Location to store text mode labels
    is_cls = True,
    clear_cache=True,
    model_name='bert-base-multilingual-cased' ## Language model
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
    model, data, "/data/models/LRL_multilingual-cased_mbert.cpt", ## Location to store the model
    t_total=num_epochs * len(data.train_dl))

model.get_n_trainable_params()

learner.fit(epochs=num_epochs)
