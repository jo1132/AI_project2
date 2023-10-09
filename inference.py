import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='get arguments')

    parser.add_argument(
        '--image_path',
        default='/root/inference_data',
        type=str,
        required=False,
        help='inference data path'
    )
    parser.add_argument(
        '--model_path',
        default='/root/Train_Model.h5',
        type=str,
        required=False,
        help='model path'
    )

    args = parser.parse_args()
    return args

# 필요한 라이브러리들을 임포트합니다.
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import os


args = parse_args()

labels = []

# 모델을 로드합니다.
model = load_model(args.model_path)
# 모델 요약
print(model.summary())

# 데이터를 로드합니다.
print('load data')
print('image_path:', args.image_path)
for file_name in os.listdir(args.image_path):
    path = os.path.join(args.image_path, file_name)
    image = Image.open(path).convert("L")
    image.thumbnail((28, 28),  Image.NEAREST)
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(-1, 28, 28)

    # 입력 데이터를 [0, 1] 범위로 정규화합니다.
    image = image.astype('float32') / 255.0

    # 모델 예측
    pred = model.predict(image)
    pred = np.argmax(pred)

    labels.append(pred)

df = pd.DataFrame(labels)
df.to_csv('inference.csv', index=False)

# 모델 추론 결과 저장
print("inference.csv 저장")

