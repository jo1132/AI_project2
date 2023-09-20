import argparse

import sklearn
from sklearn.metrics import accuracy_score

import pandas as pd
import random

def parse_args():
    parser = argparse.ArgumentParser(description='get arguments')

    parser.add_argument(
        '--label_path',
        default='/root/scikit_learn_data/test/labels.csv',
        type=str,
        required=False,
        help='scoring label path'
    )

    parser.add_argument(
        '--pred_path',
        default='/root/scikit_learn_data/test/labels.csv',
        type=str,
        required=False,
        help='predicted label path'
    )

    parser.add_argument(
        '--random',
        default=False,
        type=bool,
        required=False,
        help='predict as random',
    )

    parser.add_argument(
        '--demo',
        default=False,
        type=bool,
        required=False,
        help='For Demo',
    )


    args = parser.parse_args()
    return args


args = parse_args()
label_path = args.label_path
labels = pd.read_csv(label_path)
labels = labels.loc[:, '1']

if args.random:
    num = [i for i in range(10)]
    pred = [random.choice(num) for _ in range(len(labels))]

elif (args.demo):
    pred_path = args.pred_path
    pred = pd.read_csv(pred_path)
    pred = pred.loc[:, ['1']]

else:
    pred_path = args.pred_path
    pred = pd.read_csv(pred_path)

acc = accuracy_score(pred, labels)
print('accuracy:', acc)