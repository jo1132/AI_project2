import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='get arguments')

    parser.add_argument(
        '--image_path',
        default='./dataset/midterm/Validation',
        type=str,
        required=False,
        help='inference data path'
    )
    parser.add_argument(
        '--model_path',
        default='./Train_Model.h5',
        type=str,
        required=False,
        help='model path'
    )

    args = parser.parse_args()
    return args

# 필요한 라이브러리들을 임포트합니다.
import os
import cv2
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence

class Dataloader(Sequence):
    def __init__(self, base_dataset_path, images, batch_size):
        self.base_dataset_path = base_dataset_path
        self.images = images
        self.batch_size = batch_size
        self.indices = np.arange(len(self.images))

    def __len__(self):
        return math.ceil(len(self.images)/self.batch_size)
    
    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_x = [self.images[i] for i in indices]
        batch_images = self.get_imagesets(batch_x)
        batch_images = batch_images.astype('float32') / 255.0
        return np.array(batch_images)
                                                    
    def get_imagesets(self, path_list):
        image_list = []
        for image in path_list:

            image_path = os.path.join(self.base_dataset_path, image)
            image_list.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
        return np.array(image_list)

args = parse_args()

image_names = []
labels = []

# 모델을 로드합니다.
model = load_model(args.model_path)
# 모델 요약
print(model.summary())

# 데이터를 로드합니다.
print('load data')
print('image_path:', args.image_path)

df = pd.DataFrame(os.listdir(args.image_path), columns=['image'])
inference_dataloader = Dataloader(args.image_path, df['image'].values, 64)

# 모델 예측
pred = model.predict(inference_dataloader)

pred = list(map(np.argmax, pred))
df['label'] = pred

df.to_csv('inference.csv', index=False)

# 모델 추론 결과 저장
print("inference.csv 저장")

