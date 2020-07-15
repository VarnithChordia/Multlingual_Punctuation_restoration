from modules import tqdm
from sklearn_crfsuite.metrics import flat_classification_report
import logging
import torch
from .optimization import BertAdam
from modules.analyze_utils.plot_metrics import get_mean_max_metric
from modules.data.bert_data_clf import get_data_loader_for_predict


def train_step(dl, model, optimizer, num_epoch=1):
    model.train()
    epoch_loss = 0
    idx = 0
    pr = tqdm(dl, total=len(dl), leave=False)
    for batch in pr:
        idx += 1
        model.zero_grad()
        loss = model.score(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.data.cpu().tolist()
        epoch_loss += loss
        pr.set_description("train loss: {}".format(epoch_loss / idx))
        torch.cuda.empty_cache()
    logging.info("\nepoch {}, average train epoch loss={:.5}\n".format(
        num_epoch, epoch_loss / idx))


def transformed_result_cls(preds, target_all, cls2label, return_target=True):
    preds_cpu = []
    targets_cpu = []
    for batch_p, batch_t in zip(preds, target_all):
        for pred, true_ in zip(batch_p, batch_t):
            preds_cpu.append(cls2label[pred.cpu().data.tolist()])
            if return_target:
                targets_cpu.append(cls2label[true_.cpu().data.tolist()])
    if return_target:
        return preds_cpu, targets_cpu
    return preds_cpu


def validate_step(dl, model, id2cls):
    model.eval()
    idx = 0
    preds_cpu_cls, targets_cpu_cls = [], []
    for batch in tqdm(dl, total=len(dl), leave=False, desc="Validation"):
        idx += 1
        preds_cls = model.forward(batch)
        preds_cpu_, targets_cpu_ = transformed_result_cls([preds_cls], [batch[-1]], id2cls)
        preds_cpu_cls.extend(preds_cpu_)
        targets_cpu_cls.extend(targets_cpu_)
    clf_report_cls = flat_classification_report([targets_cpu_cls], [preds_cpu_cls], digits=4)
    return clf_report_cls


def predict(dl, model, id2cls):
    model.eval()
    idx = 0
    preds_cpu_cls = []
    for batch in tqdm(dl, total=len(dl), leave=False, desc="Predicting"):
        idx += 1
        preds_cls = model.forward(batch)
        preds_cpu_ = transformed_result_cls([preds_cls], [preds_cls], id2cls, False)
        preds_cpu_cls.extend(preds_cpu_)

    return preds_cpu_cls


class NerLearner(object):

    def __init__(self, model, data, best_model_path, lr=0.001, betas=[0.8, 0.9], clip=1.0,
                 verbose=True, t_total=-1, warmup=0.1, weight_decay=0.01,
                 validate_every=1, schedule="warmup_linear", e=1e-6):
        logging.basicConfig(level=logging.INFO)
        self.model = model
        self.optimizer = BertAdam(model, lr, t_total=t_total, b1=betas[0], b2=betas[1], max_grad_norm=clip)
        self.optimizer_defaults = dict(
            model=model, lr=lr, warmup=warmup, t_total=t_total, schedule=schedule,
            b1=betas[0], b2=betas[1], e=e, weight_decay=weight_decay,
            max_grad_norm=clip)

        self.lr = lr
        self.betas = betas
        self.clip = clip
        self.t_total = t_total
        self.warmup = warmup
        self.weight_decay = weight_decay
        self.validate_every = validate_every
        self.schedule = schedule
        self.data = data
        self.e = e
        self.best_model_path = best_model_path
        self.verbose = verbose
        self.cls_history = []
        self.epoch = 0
        self.best_target_metric = 0.

    def fit(self, epochs=100, resume_history=True, target_metric="f1"):
        if not resume_history:
            self.optimizer_defaults["t_total"] = epochs * len(self.data.train_dl)
            self.optimizer = BertAdam(**self.optimizer_defaults)
            self.cls_history = []
            self.epoch = 0
            self.best_target_metric = 0.
        elif self.verbose:
            logging.info("Resuming train... Current epoch {}.".format(self.epoch))
        try:
            for _ in range(epochs):
                self.epoch += 1
                self.fit_one_cycle(self.epoch, target_metric)
        except KeyboardInterrupt:
            pass

    def fit_one_cycle(self, epoch, target_metric="f1"):
        train_step(self.data.train_dl, self.model, self.optimizer, epoch)
        if epoch % self.validate_every == 0:
            rep_cls = validate_step(self.data.valid_dl, self.model, self.data.train_ds.idx2cls)
            self.cls_history.append(rep_cls)
        idx, metric = get_mean_max_metric(self.cls_history, target_metric, True)
        if self.verbose:
            logging.info("on epoch {} by max_{}: {}".format(idx, target_metric, metric))
            print(self.cls_history[-1])

        # Store best model
        if self.best_target_metric < metric:
            self.best_target_metric = metric
            if self.verbose:
                logging.info("Saving new best model...")
            self.save_model()

    def predict(self, dl=None, df_path=None, df=None):
        if dl is None:
            dl, ds = get_data_loader_for_predict(self.data, df_path, df)
        return predict(dl, self.model, self.data.train_ds.idx2cls)
    
    def save_model(self, path=None):
        path = path if path else self.best_model_path
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path=None):
        path = path if path else self.best_model_path
        self.model.load_state_dict(torch.load(path)).to('cpu')
