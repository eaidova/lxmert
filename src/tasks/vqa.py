# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VizWizVQADataset, VQATorchDataset, VQAEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
datasets = {
    'vqa': VQADataset,
    'vizwiz': VizWizVQADataset
}


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False, dataset_type='vqa', encoder_type='lxrt') -> DataTuple:
    dataset_class = datasets.get(dataset_type)
    if dataset_class is None:
        raise ValueError("unsupported dataset type {}".format(dataset_type))

    dset = dataset_class(splits, args.vocab_size)
    tset = VQATorchDataset(dset, encoder_type)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True, dataset_type=args.dataset_type
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=128,
                shuffle=False, drop_last=False,
                dataset_type=args.dataset_type
            )
        else:
            self.valid_tuple = None
        # Model
        self.model = VQAModel(
            self.train_tuple.dataset.num_answers if not args.transfer_learning else VQADataset.get_answers_number(),
            encoder_type = args.encoder_type
        )

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        self.prepare_model()
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def prepare_model(self):
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                scores = nn.functional.softmax(logit, dim=1).cpu().detach().numpy()

                score, label = logit.max(1)
                b = 0
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    if not isinstance(qid, str):
                        quesid2ans[qid.item()] = (ans, scores[b, l])
                    else:
                        quesid2ans[qid] = (ans, scores[b, l])
                    b += 1

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None, dataset_type='vqa'):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                scores = nn.functional.softmax(logit, dim=1).cpu().numpy()
                score, label = logit.max(1)
                b = 0
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    if not isinstance(qid, str):
                        quesid2ans[qid.item()] = (ans, scores[b, l])
                    else:
                        quesid2ans[qid] = (ans, scores[b, l])
                    b += 1
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump, dataset_type)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(
            quesid2ans, args.dump_errors, args.dump_with_score is not None,
            self.output+'/errors', args.images_dir,
            self.output + '/score_{}'.format(args.dump_with_score), args.dump_with_score)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                if not isinstance(qid, str):
                    qid= qid.item()
                quesid2ans[qid] = (ans, 1)
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)
        if args.transfer_learning:
            self.model.create_head(self.train_tuple.dataset.num_answers)
            self.prepare_model()


if __name__ == "__main__":
    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False, dataset_type=args.dataset_type),
                dump=os.path.join(args.output, 'test_predict.json'), dataset_type=args.dataset_type
            )
        elif 'val' in args.test:
            # Since part of validation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('val', bs=950,
                               shuffle=False, drop_last=False, dataset_type=args.dataset_type),
                dump=os.path.join(args.output, 'val_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)
