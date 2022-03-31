# 아주 간단한 개념입니다.
# train할때 224로 resize할겁니다.
# 그럼 매 epoch마다 하는 것은 비효율적이지요
# 따라서 아예 224 size and npy로 미리 저장해 두는것이 가장 빠릅니다.
# 하지만 test set이 어떤것이 주어질지 모르니 image file로 저장합니다.
# 추가로 test/ val로 나누며 파일 이름 마지막에는 class를 줍니다.  


#### 우선 cifar100.py를 실행하여 image를 만들어야 합니다. ###

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='CIFAR100 다운받은 위치 /cifar-100-python을 입력하세요')   # STL_10 dataset을 저장할 dir를 받습니다.
    return parser.parse_args()

def rename_resize(data_path_, train_test):

    data_path = data_path_ + "/" + str(train_test) + "_image"
    class_list  = os.listdir(data_path)     # class 목록입니다.

    if not os.path.isdir(data_path_ + "/" + str(train_test) + "_resized"):  # 존재한다면
        os.mkdir(data_path_ + "/" + str(train_test) + "_resized")

    for class_, i in tqdm(enumerate(class_list)):  # 같은 class는 같은 이름의 마지막을 같게 합니다.
        path = data_path + "/" + str(i)
        class_list = os.listdir(path)   # class별로 안에 있는 images.

        for name_, j in enumerate(class_list):
            image_path = path + "/" + str(j) # 원래 있던 images.
            image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2는 이렇게 바꿔줘야 하는건 맞는데 다시 저장하니까 하면 안된다.
            image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_CUBIC)   # 선형보간법
            
            if int(class_) <10 :    # 나중에 target을 filename에서 slicing 할건데 이렇게 미리 해두는 것이 편하다.
                class_pad = "0" + str(class_)
                new_path = data_path_ + "/" + str(train_test) + "_resized" + "/" + str(name_) + "_" + str(class_pad) + ".png"
                cv2.imwrite(new_path, image) # 저장
            else:
                new_path = data_path_ + "/" + str(train_test) + "_resized" + "/" + str(name_) + "_" + str(class_) + ".png"
                cv2.imwrite(new_path, image) 

if __name__ == "__main__":
    args = args()

    for train_test in ["train","test"]:
        print(f"resizing..... {train_test}")
        rename_resize(args.dir, train_test)
        print(f"resized !! {train_test}")
        print(f"Done!!!!!! check resized renamed datas in {args.dir}/{train_test}_resized")
# python preprocessing.py --dir /home/mskang/hyeokjong/ResNet/cifar-100-python