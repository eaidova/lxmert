# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
from collections import Counter
import re

import cv2
from PIL import Image, ImageDraw, ImageFont
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
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat'
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
    def __init__(self, splits: str, vocab_size=5000):
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

        ans2label_file = "data/vizwiz/trainval_ans2label_{}.json".format(vocab_size)
        label2ans_file = "data/vizwiz/trainval_label2ans_{}.json".format(vocab_size)
        if os.path.exists(ans2label_file) and os.path.exists(label2ans_file):
            # Answers
            with open(ans2label_file) as ans2label_file_:
                self.ans2label = json.load(ans2label_file_, object_hook=jsonValues2int)
            with open(label2ans_file) as label2ans_file_:
                self.label2ans = json.load(label2ans_file_, object_hook=jsonKeys2int)
        else:
            self.ans2label, self.label2ans = self.create_answer_vocab(top_k=vocab_size)
            print("Answer vocab with {} prepared.".format(len(self.ans2label)))
            print("Vocab will be saved to data/vizwiz directory")
            with open(ans2label_file, "w") as ans2label_file_:
                json.dump(self.ans2label, ans2label_file_)
            with open(label2ans_file, "w") as label2ans_file_:
                json.dump(self.label2ans, label2ans_file_)
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
            datum['sent'] = datum['question']
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
            # if answer in ['unsuitable', 'unsuitable image', 'unanswerable', 'too blurry']:
            #     prepared_sample_answers.append('unanswerable')
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
    def __init__(self, splits: str, *args, **kwargs):
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
    def __init__(self, dataset: VQADataset, encoder_type='lxrt'):
        super().__init__()
        self.raw_dataset = dataset
        self.encoder_type = encoder_type

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
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:, :4] = boxes
        image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) * (
                image_location[:, 2] - image_location[:, 0]) / (float(img_w) * float(img_h))

        image_location[:, 0] = image_location[:, 0] / float(img_w)
        image_location[:, 1] = image_location[:, 1] / float(img_h)
        image_location[:, 2] = image_location[:, 2] / float(img_w)
        image_location[:, 3] = image_location[:, 3] / float(img_h)
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
            return ques_id, feats, image_location if self.encoder_type == 'inter_bert' else boxes, ques, target
        else:
            return ques_id, feats, image_location if self.encoder_type == 'inter_bert' else boxes, ques


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict, visualize_errors=False, visualize_high_score=False, save_err_dir='errors', images_dir='vizwiz/images/val', save_img_dir='high_score', score_threshold=0.5):
        if visualize_errors and not os.path.exists(save_err_dir):
            os.mkdir(save_err_dir)
        if visualize_high_score and not os.path.exists(save_img_dir):
            os.mkdir(save_img_dir)
        score = 0.
        for quesid, (ans, prob) in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
            elif visualize_errors:
                save_result(save_err_dir, images_dir, datum, ans, prob)
            if visualize_high_score and prob > score_threshold:
                save_result(save_img_dir, images_dir, datum, ans, prob)
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
            for ques_id, (ans, prob) in quesid2ans.items():
                sample = {'answer': ans}
                if dataset_type == 'vizwiz':
                    image_id = '{}.jpg'.format(ques_id)
                    sample['image'] = image_id
                else:
                    sample['question_id'] = ques_id
                result.append(sample)
            json.dump(result, f, indent=4, sort_keys=True)


def save_result(save_dir, images_dir, datum, answer=None, answer_score=None):
    img_id = datum['img_id']
    ques = datum['sent']
    label = datum['label']
    image_path = os.path.join(images_dir, '{}.jpg'.format(img_id))
    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]
    text = ['Q: {}'.format(ques),
            'gt answers: {}'.format(', '.join(['{} - {:.2}'.format(l, float(score)) for l, score in label.items()]))]
    if answer:
        text.append('pred answer: {} {}'.format(
            answer, '- {:.6}'.format(answer_score) if answer_score is not None else ''))
    printing_lines = []
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    line_height = font.getsize('hg')[1]

    for t in text:
        printing_lines.extend(text_wrap(t, font, img_w))

    text_h = len(printing_lines) * line_height + 20
    text_wind = np.full((text_h, img_w, 3), 255, dtype=np.uint8)
    txt = Image.fromarray(text_wind)
    draw = ImageDraw.Draw(txt)
    x = 10
    y = 20
    for line in printing_lines:
            # draw the line on the image
        draw.text((x, y), line, fill=0, font=font)
        y = y + line_height
    image = cv2.vconcat([np.array(txt), image])
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format(img_id)), image)


def text_wrap(text, font, max_width):
    lines = []
    # If the width of the text is smaller than image width
    # we don't need to split it, just add it to the lines array
    # and return
    if font.getsize(text)[0] <= max_width:
        lines.append(text)
    else:
        # split the line by spaces to get words
        words = text.split(' ')
        i = 0
        # append every word to a line while its width is shorter than image width
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + " "
                i += 1
            if not line:
                line = words[i]
                i += 1
            # when the line gets longer than the max width do not append the word,
            # add the line to the lines array
            lines.append(line)
    return lines
