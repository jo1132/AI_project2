import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='get arguments')

    parser.add_argument(
        '--dataset_path',
        default='./dataset/midterm/Training',
        type=str,
        required=False,
        help='dataset path'
    )
    parser.add_argument(
        '--label_path',
        default='./dataset/midterm/Training/label.csv',
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
        default='Train_Model',
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
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence

class Dataloader(Sequence):

    def __init__(self, dataset_path, label_path):
        self.dataset_path = dataset_path 
        self.label_path = label_path
    
    def get_imagesets(self, path_list):
        basepath = self.dataset_path
        image_list = []

        for image in path_list:
            image_path = os.path.join(basepath, image)
            image_list.append(plt.imread(image_path))

        return np.rarray(image_list)


    def get_datasets(self):
        df = pd.read_csv(self.label_path)
        
        X_train_path, Y_train, x_test_path, y_test = train_test_split(df['image'], df['label'], train_size=0.8, shuffle=True)

        X_train = self.get_imagesets(X_train_path)
        x_test = self.get_imagesets(x_test_path)

        return (X_train, Y_train, x_test, y_test)


args = parse_args()
# 데이터를 로드합니다.
print('load data')

datasets = Dataloader(args.dataset_path, args.label_path)
X_train, Y_train, x_test, y_test = datasets.get_datasets()

# 입력 데이터를 [0, 1] 범위로 정규화합니다.
X_train = X_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 딥력데이터의 모양을 바꿉니다.
X_train = X_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 레이블을 one-hot 인코딩합니다.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)



# 모델을 정의합니다.
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
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
model.fit(X_train, Y_train, epochs=args.epoch, batch_size=64, validation_data=(x_test, y_test))

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {test_acc}")

# 모델 저장
model.save(args.model_name+'.h5')