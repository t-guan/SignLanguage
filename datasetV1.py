# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 23:36:00 2022

@author: Thomas Guan
"""
##Import Libraries
#We are going to preprocess all of the data, which are pixel-based data encoded to a letter, so need to perform the encoding here.
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from typing import List
import torch.nn as nn
import numpy as np
import csv
import torchvision
import torchvision.transforms as transforms

class SignLanguageMNIST(Dataset):
    
    @staticmethod
    def label_map():
        mapping=list(range(25))#0-25, total 26 letters
        mapping.pop(9)#Remove letter j because its not static, 25 letters
        return mapping#Map the labels 0-23 indexes to 0-25 letters.
    
    @staticmethod
    def read_csv(path:str):
        mapping = SignLanguageMNIST.label_map()
        labels,samples=[],[]
        with open(path) as f:
            _=next(f) #Skip Header
            for line in csv.reader(f):
                label=int(line[0]) #label, so the letter in first line
                labels.append(mapping.index(label))
                samples.append(list(map(int,line[1:])))
            return labels,samples
        
    def __init__(self,
                 path: str="data/sign_mnist_train.csv",
                 mean: List[float]=[0.485],
                 std: List[float]=[0.229]):
        labels,samples=SignLanguageMNIST.read_csv(path)
        self._samples=np.array(samples,dtype=np.uint8).reshape((-1,28,28,1))
        self._labels=np.array(labels, dtype=np.uint8).reshape((-1,1))
        
        self._mean=mean
        self._std=std
        
    def __len__(self):
        return len(self._labels)
    
    def __getitem__(self,idx):
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(28, scale=(0.8,1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean,std=self._std)])
        return{
            'image':transform(self._samples[idx]).float(),
            'label':torch.from_numpy(self._labels[idx]).float()
            }
def training_test_loaders(batch_size=32):
    trainset=SignLanguageMNIST('data/sign_mnist_train.csv')
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True)
    
    testset=SignLanguageMNIST('data/sign_mnist_test.csv')
    testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False)
    return trainloader, testloader
if __name__=='__main__':
    loader, _ = training_test_loaders(2)
    print(next(iter(loader)))
    
        
    
    
    
