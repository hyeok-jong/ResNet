# Implementation ResNet  

This repository is implementation of ResNet.  


## 1. Dataset  

I used CIFAR-100 Dataset for classification.

The dataset has 100 classes

The original data consists of 60,000 images for 50,000 for train and 10,000 for test.  

Because there are several difference with valid set and test set.  
So I split in two with same ratio.  

CIFAR-100 dataset citation [Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)  

Download dataset : [Here](https://www.cs.toronto.edu/~kriz/cifar.html)

From link above, download `CIFAR-100 python version	161 MB	eb9058c3a382ffc7106e4002c42a8d85` then file named `cifar-100-python.tar.gz` will be downloaded.  

Unzip it you could get four files named `file.txt~` , `meta` , `test` , `train`.  

It is recommended that the files be under directory `/cifar-100-python`.  

Then by using `cifar-100-python/cifar100.py` images will be generated.  

`cifar-100-python/preprocessing.py` will give you preprocessed images which are resized 224 and label will be attached in filename.  

Although those work requires more memory, training time will decrease.  
( Even more so when training more times. )

Finally `ResNet/dataloader.py` will makes `torch.DataLoader`.  

## 2. ResNet  

[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)  
[Paper reviews](https://hyeok-jong.github.io/paper%20base%20line/Paper_ResNet/)  

For visualizing model architecture `ResNet/resnet34.py` will give you onnx file.  
In [Netron](https://netron.app/) model will opend.  

## 3. Train  
In `ResNet/results` many file will be made.  


`python train.py --data_dir /home/mskang/hyeokjong/ResNet/cifar-100-python --result_dir /home/mskang/hyeokjong/ResNet/results --GPU 1 --batch_size 256 --epoch 6`








