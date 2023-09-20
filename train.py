import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='get arguments')

    parser.add_argument(
        '--dataset_path',
        default='/root/scikit_learn_data',
        type=str,
        required=False,
        help='dataset path'
    )
    parser.add_argument(
        '--label_path',
        default='labels.csv',
        type=str,
        required=False,
        help='label path'
    )
    parser.add_argument(
        '--epoch',
        default=100,
        type=int,
        required=False,
        help='Set epochs'
    )
    parser.add_argument(
        '--model_name',
        default='MNIST_model',
        type=str,
        required=False,
        help='Set model name'
    )
    args = parser.parse_args()
    return args

# 필요한 라이브러리들을 임포트합니다.
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence

class Dataloader(Sequence):

    def __init__(self, dataset_path, label_path):
        self.dataset_path = dataset_path 
        self.label_path = label_path

    def load_labels(self, path):
        path = os.path.join(path, 'labels.csv')
        return pd.read_csv(path)
    
    def get_datasets(self, file='train'):
        path = os.path.join(self.dataset_path, file)
        df = self.load_labels(path)
        
        datasets = []
        labels = []
        for image, label in df.values:
            image_path = os.path.join(path, image)
            datasets.append(plt.imread(image_path))

            labels.append(label)

        datasets = np.array(datasets)
        return (datasets, labels)

args = parse_args()
# 데이터를 로드합니다.
print('load data')
datasets = Dataloader(args.dataset_path, args.label_path)

x_train, y_train = datasets.get_datasets('train')
print(len(x_train))
x_test, y_test = datasets.get_datasets('test')
print(len(x_test))

# 입력 데이터를 [0, 1] 범위로 정규화합니다.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 딥력데이터의 모양을 바꿉니다.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 레이블을 one-hot 인코딩합니다.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)



# 모델을 정의합니다.
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 요약
print(model.summary())

# 모델 학습
print('train_model')
model.fit(x_train, y_train, epochs=args.epoch, batch_size=64, validation_data=(x_test, y_test))

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {test_acc}")

# 모델 저장
model.save(args.model_name+'.h5')