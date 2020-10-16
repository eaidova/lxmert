# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
from collections import Counter
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = 'data/vqa/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
SPLIT2NAME = {
    'train': 'train',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test',
    'val': 'val'
}
vizwiz_splits = {
    'train': 'train',
    'valid': 'val',
    'test': 'test',
    'val': 'val'
}


class VizWizVQADataset:
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(self.preprocess_data(json.load(open("data/vizwiz/Annotations/{}.json".format(vizwiz_splits[split]))), False))
        print("Load {} data from split(s) {}.".format(len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        def jsonKeys2int(x):
            if isinstance(x, dict):
                return {int(k): v for k, v in x.items()}
            return x

        def jsonValues2int(x):
            if isinstance(x, dict):
                return {k: int(v) for k, v in x.items()}
            return x

        if os.path.exists("data/vizwiz/trainval_ans2label.json") and os.path.exists("data/vizwiz/trainval_label2ans.json"):
            # Answers
            with open("data/vizwiz/trainval_ans2label.json") as ans2label_file:
                self.ans2label = json.load(ans2label_file, object_hook=jsonValues2int)
            with open("data/vizwiz/trainval_label2ans.json") as label2ans_file:
                self.label2ans = json.load(label2ans_file, object_hook=jsonKeys2int)
        else:
            self.ans2label, self.label2ans = self.create_answer_vocab(top_k=5000)
            print("Answer vocab with {} prepared.".format(len(self.ans2label)))
            print("Vocab will be saved to data/vizwiz directory")
            with open("data/vizwiz/trainval_ans2label.json", "w") as ans2label_file:
                json.dump(self.ans2label, ans2label_file)
            with open("data/vizwiz/trainval_label2ans.json", "w") as label2ans_file:
                json.dump(self.label2ans, label2ans_file)
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)

    def preprocess_data(self, raw_dataset, filter_unanswerable=True):
        preprocessed_data = []
        for datum in raw_dataset:
            image_id = datum['image'].rsplit('.')[0]
            datum['img_id'] = image_id
            datum['question_id'] = image_id
            datum['sent'] = self.preprocess_question(datum['question'])
            if 'answers' in datum:
                if filter_unanswerable and not datum['answerable']:
                    continue
                answers = datum['answers']
                count_answ = Counter(self.prepare_answer([ans['answer'] for ans in answers]))
                label = {answer: min(1, score / 3) for answer, score in count_answ.items()}
                datum['label'] = label
            preprocessed_data.append(datum)
        return preprocessed_data

    @staticmethod
    def prepare_answer(answers):
        prepared_sample_answers = []
        for answer in answers:
            answer = answer.lower()

            # define desired replacements here
            punctuation_dict = {'.': ' ', "'": '', '?': ' ', '_': ' ', '-': ' ', '/': ' ', ',': ' '}

            rep = punctuation_dict
            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile("|".join(rep.keys()))
            answer = pattern.sub(lambda m: rep[re.escape(m.group(0))], answer)
            prepared_sample_answers.append(answer)

        return prepared_sample_answers

    @staticmethod
    def preprocess_question(question):
        conversational_dict = {"thank you": '', "thanks": '', "thank": '', "please": '', "hello": '',
                               "hi ": ' ', "hey ": ' ', "good morning": '', "good afternoon": '', "have a nice day": '',
                               "okay": '', "goodbye": ''}


        # use these three lines to do the replacement
        rep = dict((re.escape(k), v) for k, v in conversational_dict.items())
        pattern = re.compile("|".join(rep.keys()))
        question = pattern.sub(lambda m: rep[re.escape(m.group(0))], question)
        return question

    def create_answer_vocab(self, top_k=5000):
        answers = []
        for split in ['train', 'val']:
            data = json.load(open("data/vizwiz/Annotations/{}.json".format(split)))
            for obj in data:
                for ann in obj['answers']:
                    answers.extend(self.prepare_answer([ann['answer']]))

        counter = Counter(answers)
        counted_ans = counter.most_common(top_k)
        # start from labels from 0
        ans2label, label2ans = {}, {}
        for i, t in enumerate(counted_ans):
            ans2label[t[0]] = i
            label2ans[i] = t[0]

        return ans2label, label2ans


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/vqa/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        def jsonKeys2int(x):
            if isinstance(x, dict):
                return {int(k): v for k, v in x.items()}
            return x
        def jsonValues2int(x):
            if isinstance(x, dict):
                return {k: int(v) for k, v in x.items()}
            return x


        # Answers
        self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"), object_hook=jsonValues2int)
        self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"), object_hook=jsonKeys2int)
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    @staticmethod
    def get_answers_number():
        return len(json.load(open("data/vqa/trainval_ans2label.json")))

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []
        for split in dataset.splits:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            load_topk = 5000 if (split == 'minival' and topk is None) else topk
            img_data.extend(load_obj_tsv(
                os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path, dataset_type='vqa'):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                sample = {'answer': ans}
                if dataset_type == 'vizwiz':
                    image_id = '{}.jpg'.format(ques_id)
                    sample['image'] = image_id
                else:
                    sample['question_id'] = ques_id
                result.append(sample)
            json.dump(result, f, indent=4, sort_keys=True)
