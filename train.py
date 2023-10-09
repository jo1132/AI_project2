import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='get arguments')

    parser.add_argument(
        '--dataset_path',
        default='./dataset/midterm/Training',
        #default='./dataset/midterm/Validation',
        type=str,
        required=False,
        help='dataset path'
    )
    parser.add_argument(
        '--label_path',
        default='./dataset/midterm/Training/label.csv',
        #default='./dataset/midterm/label.csv',
        type=str,
        required=False,
        help='label path'
    )
    parser.add_argument(
        '--epoch',
        default=10,
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
import cv2
import os
import tensorflow as tf
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence

class Dataloader(Sequence):
    def __init__(self, base_dataset_path, images, labels, batch_size):
        self.base_dataset_path = base_dataset_path
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.indices = np.arange(len(self.labels))

    def __len__(self):
        return math.ceil(len(self.labels)/self.batch_size)
    
    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_x = [self.images[i] for i in indices]
        batch_images = self.get_imagesets(batch_x)
        batch_images = batch_images.astype('float32') / 255.0

        batch_y = [self.labels[i] for i in indices]
        batch_y = to_categorical(batch_y, 10)
        return np.array(batch_images), np.array(batch_y)
    
    def get_imagesets(self, path_list):
        image_list = []
        for image in path_list:
            image_path = os.path.join(self.base_dataset_path, image)
            image_list.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
        return np.array(image_list)



args = parse_args()

# 데이터를 로드합니다.
print('load data')
df = pd.read_csv(args.label_path)
X_train_path, x_test_path, Y_train, y_test = train_test_split(df['image'].values, df['label'].values, train_size=0.8, shuffle=True, stratify=df['label'])

train_loader = Dataloader(args.dataset_path, X_train_path, Y_train, batch_size=64)
test_loader = Dataloader(args.dataset_path, x_test_path, y_test, batch_size=64)

# 모델 생성
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')  # num_classes에는 종류별 클래스 수를 설정하세요.
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 요약
print(model.summary())

# 모델 학습
print('train_model')
model.fit(train_loader, validation_data=test_loader, epochs=args.epoch, batch_size=64)#, validation_data=(x_test, y_test))

# 모델 평가
test_loss, test_acc = model.evaluate(test_loader)
print(f"테스트 정확도: {test_acc}")

# 모델 저장
model.save(args.model_name+'.h5')