
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
# import utils
import time
import re
import copy
import os
# import model
from PIL import Image
seed = 2014
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
import random
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import sys
sys.path.append("..")
import utils

class DataAugment():
    def __init__(self):
        self.num_workers = 0
        self.aug_num = 15
        self.transform = transforms.Compose([transforms.ToTensor()])
    def dataload_for_train(self):
        print('Training data augmentation ...')
        datasetData = ImageFolder('./Data_train',transform=self.transform)
        datasetLabel = ImageFolder('./Label_train',transform=self.transform)        
        dataloaderData = DataLoader(datasetData, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=True)  
        dataloaderLabel = DataLoader(datasetLabel, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=True)  
        dataTrain = zip(dataloaderData,dataloaderLabel)   
        for batch,loadTrain in enumerate(dataTrain):                  
            data0, label0 = loadTrain[0][0], loadTrain[1][0]
            for i in range(self.aug_num):
                data_i, label_i = utils.get_patch(data0, label0)# ([batchsize, 3, patch, patch])
                print(batch,i)
                # print(data_i.shape)  # (1,3,128,128)
                data_i, label_i = data_i.numpy()*255, label_i.numpy()*255
                h_data = Image.fromarray((np.squeeze(data_i)).transpose(1,2,0).astype(np.uint8))
                h_label = Image.fromarray((np.squeeze(label_i)).transpose(1,2,0).astype(np.uint8))
                h_data.save('./Data_train_aug/Images/data_'+str(batch+1)+'_'+str(i+1)+'.png')
                h_label.save('./Label_train_aug/Images/label_'+str(batch+1)+'_'+str(i+1)+'.png')
        
        print('Training data augmentation is over.')
        # return dataTrain, len(dataloaderData)
        
        
dA = DataAugment()
dA.dataload_for_train()
        
        
        
        
        
        
        
        