# Dataset과 transformation은 직접 만듭니다.
# Dataset과 DataLoader는 다릅니다.
# Dataset은 image의 tranformation을 진행하는 부분이고
# Dataloader는 Dataset에 의해 변환된 image를 batch_szie로 묶고 shuffle하고 nem_workers, pin_memory를 지정합니다.
# 따라서 transformation은 Dataset에 있고 Bathc_size는 DataLoader에 있습니다.  
# 또한 Torch의 transformation은 순서가 다르면 안되고 정확히는 tensor인지 PIL인지 Numpy인지가 일관적이지가 않습니다.
# 어째든 결론은 Image file을 최종적으로 tesnor로 바꿔줘야 하는데 중간의 transformation이 있습니다.  
# 수많은 실험을 통해서 최적의 방법은 cv2로 읽어 numpy로 변환하고 cv2를 이용하여 numpy상태로 transformation을 진행하여  
# 최종적으로 tensor로 바꿔주는 것이라는 것을 찾았습니다.
# https://velog.io/@jj770206/Split-dataset를 참고하세요

import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import argparse

class custom_dataset(Dataset):
    def __init__(self, input_dir, transform = None):   # filename 자체가 class이므로 target_dir은 없습니다.
        self.input_dir = input_dir
        self.input_list = os.listdir(input_dir)        # Input image의 list입니다.
        self.transform = transform
        
    def __len__(self):
        return len(self.input_list)                    # 없어도 문제는 안될거 같지만 원래 Dataset의 form을 유지하기 위해 사용합니다.

    def __getitem__(self,idx):                         # 이부분이 Iterate되는 부분입니다.
        os.chdir(self.input_dir)                       # Train과 Test모두 이용하므로 위치를 해당 dir로 바꿔줍니다.
        input_image_numpy = cv2.imread(self.input_list[idx])     # idx번째 파일을 numpy로 읽습니다.

        # 만약 resize가 필요하다면 아래 코드 주석 없애면 됩니다. 절대 비추입니다. 
        # 미리 저장해두세요.
        #input_image_numpy = cv2.resize(input_image_numpy, (224,224), interpolation = cv2.INTER_LANCZOS4) 

        target_ = int(self.input_list[idx][-6:-4])         # filename을 slicing하여 target으로 활용합니다.
        

        # segmentation인 경우에는 아래와 같이 image와 target을 합쳐 transformation에 넣어주어 같은 함수를 지나게 해야 합니다.
        # 지금은 classification이므로 target을 같이 넣어줄 필요는 없으나 확장성을 위해 같이 넣어주고 tranformation에서 target부분은 주석ㅊ리 합니다.
        combine = {'input':input_image_numpy, 'target':target_}


        if self.transform:                             # transformation이 있으면 적용합니다. resize는 여기서 적용하는 것 보다 미리 적용하여 다시 저장하는 것이 효율적입니다.
            combine = self.transform(combine)
        
        # 즉, tranformation
        
        input_tensor = torchvision.transforms.functional.to_tensor(combine['input'])
        target_tensor = torch.tensor(combine['target'])   # target도 단순히 int가 아닌 tensor로 바꿔줍니다.
        
        return (input_tensor, target_tensor)




class RandomFlip():  # segmenstaion같은 경우는 target도 같이 바꿔야 하는데 지금은 classification 생략합니다.
    # input으로 numpy를 받는다.
    def __init__(self, horizontal = True, vertical = False, p = 0.5): 
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p # p는 그냥 예의상 넣었다. 건들이는 경우가 있나 싶긴하다
        

        
    def __call__(self, combine):   # segmentation을 위한 작업

        inputs = combine['input']   # (224, 224, 3)
        targets = combine['target']  # segmeantaion일때 주석 제거해서 같은 함수의 tranformation을 거치게 합니다.
           
        if (self.horizontal) and (np.random.rand() > self.p):
            inputs = cv2.flip(inputs,1)
            # targets = cv2.flip(targets,1)
        
        if (self.vertical) and (np.random.rand() > self.p):
            inputs = cv2.flip(inputs,0)
            # targets = cv2.flip(targets,0)

        combine = {'input': inputs, 'target': targets}   # classfication이므로 target은 그대로 입니다. 아예 생략해도 됩니다.

        return combine  



#$%^&*()(*&^%$%^&*(*&^%$# 모든 기준은 해당 dadtaset의 directory를 변경하지 않는 다는 가정하에 이루어 집니다.



def make_dataloader(dir, batch_size, transform = RandomFlip()):

    train_dir = dir + "/train_resized"
    test_dir = dir + "/test_resized"

    train_dataset = custom_dataset(train_dir, transform)
    valid_dataset = custom_dataset(test_dir)
    valid_dataset, test_dataset = split(valid_dataset, 0.5)
    
    train_dl = DataLoader(train_dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    val_dl = DataLoader(valid_dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    test_dl = DataLoader(test_dataset, batch_size, shuffle = False, num_workers = 4, pin_memory = True)

    return train_dl, val_dl, test_dl

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory where ur datasets exists. e.g) /home/mskang/hyeokjong/ResNet/cifar-100-python')   # STL_10 dataset을 저장할 dir를 받습니다.
    return parser.parse_args()




# 보통 train/ test 이렇게 주어지는데 tetest로 나눠줘야 합니다.
# 엄밀하게 말하면 test와 valid는 다릅니다.
# test는 말 그대로 trained model이 한번도 보지 못한 dataset에 대해서도 성능이 좋은가를 평가하는 역할입니다.  
# valid역시 같은 역할을 하지만 매 epoch마다 metric과 loss를 게산하여 overfitting이 일어난는 것을 확인합니다.  
# 즉, 실제 학습은 train으로 되지만 valid역시 학습에 overfitting을 방지한다는 역할을 하기 때문에 
# 엄밀하게는 test dataset으로 사용되면 안됩니다.
# 심지어 valid를 기준으로 lr을 낮추기도 하는데 이때는 더욱 그렇습니다.

# 해당 함수는 같은 directory에 test와 valid가 같이 있는 경우입니다.
# 그냥 전처리 할때 folder를 세개 만드는 것으 더 추천합니다.
# image전처리 과정에서 여기까지 생각하여 class별로 random하게 sampling되게 했는데
# uniformly sampling은 상당히 주요한 과정이라 다른 dataset을 이용한다면 
# 꼭 확인해야 합니다.
def split(dataset, ratio): # raion는 valid dataset 기준으로 합니다.
    valid_dataset_size = int(ratio*len(dataset))
    test_dataset_size = len(dataset) - valid_dataset_size
    valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [valid_dataset_size, test_dataset_size])
    return valid_dataset, test_dataset


if __name__ == "__main__":

    args = args()
    train_dir = args.dir + "/train_resized"
    test_dir = args.dir + "/test_resized"

    print("dataset test .............................................")
    train_dataset = custom_dataset(train_dir)
    for i, j in train_dataset:
        print("len of train dataset", len(train_dataset))
        print("shape and target of train dataset ",i.shape, j)
        break


    valid_dataset = custom_dataset(test_dir)
    valid_dataset, test_dataset = split(valid_dataset, 0.5)

    for i, j in valid_dataset:
        print("len of valid dataset", len(valid_dataset))
        print("shape and target of valid dataset ",i.shape, j)
        break   

    for i, j in test_dataset:
        print("len of test dataset", len(test_dataset))
        print("shape and target of test dataset ",i.shape, j)
        break 

    print("dataloader test .............................................")

    train_dl, val_dl, test_dl = make_dataloader(args.dir, batch_size = 16) 

    for i, j in train_dl:
        print("epochs train", len(train_dl))
        print("shape and target of train data ",i.shape, j)
        break
    for i, j in val_dl:
        print("epochs valid", len(val_dl))
        print("shape and target of valid data ",i.shape, j)
        break   
    for i, j in test_dl:
        print("epochs test", len(test_dl))
        print("shape and target of test data ",i.shape, j)
        break 
    ######   python dataloader.py --dir  /home/mskang/hyeokjong/ResNet/cifar-100-python  ########
