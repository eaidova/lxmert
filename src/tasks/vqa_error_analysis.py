import argparse
import json
from collections import Counter
from pathlib import Path
import cv2


def parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', required=False, default='data/vizwiz/val', help='dataset root')
    parser.add_argument('-a', '--annotation', required=False, default='data/vizwiz/Annotations/val.json', help='annotation file')
    parser.add_argument('-p', '--prediction', required=True, help='prediction file')
    return parser.parse_args()

def get_error_list(annotation_file, prediction_file):
    with open(annotation_file, 'r') as ann_file:
        annotations = json.load(ann_file)
        ann_dict = {ann['image']: ann for ann in annotations}
    with open(prediction_file, 'r') as pred_file:
        predictions = json.load(pred_file)
        pred_dict = {pred['image']: pred for pred in predictions}
    error_list = []
    for image, prediction in pred_dict.items():
        annotation = ann_dict[image]
        pred_answer = prediction['answer']
        annotation_answers = [gt['answer'] for gt in annotation['answers']]
        if pred_answer not in annotation_answers:
            answer_cnt= Counter(annotation_answers)
            answer_with_score = [(ans, min(1, score/3) for ans, score in answer_cnt.items())]
            error_list.append({
                'image': image,
                'question': annotation['question']
                'prediction_answer': pred_answer,
                'annotation_answers': answer_with_score
            })
    print('Total images {}'.format(len(ann_dict)))
    print('Errors found {}'.format(len(error_list)))
    return error_list

def prepare_result(data, data_dir):
    image_path = data_dir / data['image']
    image = cv2.imread(str(image_path))
    question = data['question']
    pred_answer = data['prediction_answer']
    gt_answers = data['annotation_answers']
    pos = (10, 50)
    pos_ans = (10, 70)
    pos_gt = (10, 90)
    image = cv2.putText(image, 'q: {}'.format(question), pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255))
    image = cv2.putText(image, 'predicted answer: {}'.format(pred_answer), pos_ans, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255))
    ans_with_score = ['{} - {}'.format(ans, score) for ans, score in gt_answers]
    image = cv2.putText(image, 'suggested answers: {}'.format(', '.join(ans_with_score)), pos_gt, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255))
    return image


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    error_list = get_error_list(args.annotation_file, args.prediction_file)
    for error_data in error_list:
        image = prepare_result(error_data, data_dir)
        cv2.imshow(error_data['image'], image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
