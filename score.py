import argparse
import sklearn
import pandas as pd
import random

def parse_args():
    parser = argparse.ArgumentParser(description='get arguments')

    parser.add_argument(
        '--label_path',
        default='./dataset/midterm/Validation/label.csv',
        type=str,
        required=False,
        help='scoring label path'
    )

    parser.add_argument(
        '--pred_path',
        default='./dataset/midterm/Validation/label.csv',
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

    args = parser.parse_args()
    return args

def  score_two_csv(pred_df, labels_df):
    score = 0
    labels = labels_df.set_index(['image'])

    pred_names = pred_df['image'].values
    for name, label in pred_df.values:
        score += 1 if labels.loc[name]['label'] == label else 0
    
    return score / len(labels_df)


args = parse_args()
label_path = args.label_path
labels = pd.read_csv(label_path)

if args.random:
    num = [i for i in range(10)] 
    pred_random = [random.choice(num) for _ in range(len(labels))]
    pred = pd.DataFrame(labels['image'], columns=['image'])
    pred['label'] = pred_random

else:
    pred_path = args.pred_path
    pred = pd.read_csv(pred_path)


acc = score_two_csv(pred, labels)
print('accuracy:', acc)