# -*- coding: utf-8 -*-
'''
@author:AsajuHuishi 
@time:  2020/10/21  authorized
'''
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
import utils
import time
import re
import copy
import os
import model
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
'''
mkdir ./log
mkdir ./test
nohup python -u main.py -gpu -train -model_name 'RCAN' -save_model_name 'RCAN' -n_resblock 8 -bt 8 > log/retroRCAN.log &
nohup python -u main.py -gpu -test -test_save_path './test/output/' -n_resblock 8 -test_model './checkpoint/RCAN/RCAN__best' > log/RCAN_test.log &
'''
# Load data
class Dataloader():
    def __init__(self,args):
        self.num_workers = 0
        self.transform = transforms.Compose([transforms.ToTensor()])
    def dataload_for_train(self,args):
        print('Training data is loading ...')
        datasetData = ImageFolder('./Data/Data_train',transform=self.transform)
        datasetLabel = ImageFolder('./Data/Label_train',transform=self.transform)        
        dataloaderData = DataLoader(datasetData, batch_size=args.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True)  
        dataloaderLabel = DataLoader(datasetLabel, batch_size=args.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True)  
        dataTrain = zip(dataloaderData,dataloaderLabel)   
        print('Loading training data load is over.')
        return dataTrain, len(dataloaderData)
    def dataload_for_valid(self,args):
        print('validating data is loading ...')
        datasetData = ImageFolder('./Data/Data_valid',transform=self.transform)
        datasetLabel = ImageFolder('./Data/Label_valid',transform=self.transform)        
        dataloaderData = DataLoader(datasetData, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=True)  
        dataloaderLabel = DataLoader(datasetLabel, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=True)  
        dataValid = zip(dataloaderData,dataloaderLabel) 
        print('Loading validating data load is over.')
        return dataValid, len(dataloaderData)        
    def dataload_for_test(self,args):
        print('testing data is loading ...')
        datasetData = ImageFolder(args.test_path,transform=self.transform)
        dataTest = DataLoader(datasetData, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=True)  
        print('Loading testing data load is over.')  
        return dataTest
# network definition
class RetroNet(nn.Module):
    def __init__(self, args):
        super(RetroNet,self).__init__()
        self.body = model.Model(args)     # now based on RCAN   
    def forward(self,x):
        x = self.body(x)
        return x 
# train/valid/test
class Train():
    def __init__(self,model,args):
        # hardware  
        self.device = args.device
        # data    
        self.d = Dataloader(args);   
        # model options
        self.model = model.to(self.device) 
        self.model_name = args.model_name
        self.save_model_name = args.save_model_name 
        # train
        self.train = args.train       
        self.start_epoch = 0   
        self.optim = args.optim
        self.continue_training = args.continue_training       
        self.pretrain_model = args.pretrain_model
        self.num_epoch = args.num_epoch
        if self.train == True:
            self.loss = args.loss  
            self.batch_size = args.batch_size          
            self.continue_training_isrequired()             
        # test
        self.test = args.test 
        if self.test == True:
            self.test_save_path = args.test_save_path
            self.test_model = args.test_model   
            self.test_name = '/VaporWave'
    # train       
    def trainer(self, args):  
        if not self.train:
            return
        if self.loss == 'L1Loss':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        self.best_output_psnr_avg,self.best_idx = 0,0
        # self.num_batch = len(list(self.vaporTrain))
        for epoch in range(self.start_epoch,self.num_epoch):  
            print('---------------------------training---------------------------')
            if self.optim == 'Adam':
                self.learning_rate = 0.0001
                optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
            elif self.optim == 'SGD':
                self.learning_rate = 0.1 ** (epoch//20 + 1)
                optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001, momentum=0.9)
            else:
                self.learning_rate =  0.0002 * 0.5 ** (epoch // 30)
                optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) if epoch<20 else optim.SGD(self.model.parameters(),lr=self.learning_rate, momentum=0.9)
            print('learning rate of this epoch {:.8f}'.format(self.learning_rate))
            self.vaporTrain, self.num_train = self.d.dataload_for_train(args) 
            for batch,loadTrain in enumerate(self.vaporTrain):                  
                data0, label0 = loadTrain[0][0], loadTrain[1][0]   
                data, label = utils.get_patch(data0, label0)# ([batchsize, 3, patch, patch])
                optimizer.zero_grad()
                data = torch.autograd.Variable(data.to(self.device),requires_grad=True)
                try:
                    output = self.model(data)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise e         
                output = output.cpu()
                total_loss = criterion(output, label)
                total_loss.backward()
                optimizer.step()                    
#                para = self.model.state_dict()
                # for i in para:
                    # print(i,para[i].size())
                # print(para)
                    # assert torch.isnan(para[i]).sum()==0, 'para has NaN!'
                del output
                if (batch) % 1 == 0: 
                    # print('==>>> epoch: {},loss10: {:.6f}'.format(epoch, loss10))
                    print('==>> epoch: {} [{}/{}]'.format(epoch+1, batch, self.num_train))
                    print('loss:%.3f'%(total_loss))
                    print(" ")
            state = {'model': self.model.state_dict(), 'epoch': epoch}
            self.ckp_path = 'checkpoint/'+self.save_model_name  # # save model path  
            if not os.path.isdir(self.ckp_path):
                os.system('mkdir -p '+self.ckp_path)        
            torch.save(state, self.ckp_path + '/'+self.model_name+'__'+str(epoch+1))
            self.validater(epoch, args)  
        print('--------------------Train over--------------------------')          
    # validate
    def validater(self, epoch, args):#epoch: now training epoch
        self.vaporValid, self.num_valid = self.d.dataload_for_valid(args)
        PSNR_output_record = np.zeros((self.num_valid))
        valid_path = './valid/'+self.save_model_name+'/'  # save validation path
        if not os.path.exists(valid_path):
            os.makedirs(valid_path)  
        for batch,load in enumerate(self.vaporValid):                  
            data1, label1 = load[0][0], load[1][0]   
            # data, label = utils.get_patch(data, label)# ([batchsize, 3, patch, patch])
            print('valid shape',data1.shape)
            data = torch.autograd.Variable(data1.to(self.device),requires_grad=False)
            label = torch.autograd.Variable(label1.to(self.device),requires_grad=False)
            with torch.no_grad():
                outputValid = self.model(data)              
            # get output PSNR
            mse_value1 = F.mse_loss(255 * outputValid, 255 * label)
            psnr1 = -10 * torch.log10(mse_value1) + torch.tensor(48.131)
            PSNR_output_record[batch] = psnr1.cpu().data.numpy()# save  PSNR output result
            del data
            print('-----epoch: %d: 第%d 张mat，output psnr: %f'%(epoch+1,batch+1,psnr1))
            print('output',outputValid.shape)  #(1, 3, 64, 64)
            # save 
            outputValid = outputValid.cpu().data.numpy()*255
            h = Image.fromarray((np.squeeze(outputValid)).transpose(1,2,0).astype(np.uint8))
            h.save(valid_path+'/valid_epoch'+str(epoch)+'_'+str(batch+1)+'.png')
            sio.savemat(valid_path+'/valid_epoch'+str(epoch)+'_'+str(batch+1)+'.mat',{'SR':np.squeeze(outputValid),'num_epoch':self.num_epoch})   
            sio.savemat(valid_path+'PSNR_output_record_'+str(epoch+1)+str(batch+1)+'.mat',{'PSNR':PSNR_output_record,'num_epoch':epoch+1})
        print('输出图像PSNR平均值：%f'%(PSNR_output_record.mean()))
        
        if PSNR_output_record.mean() > self.best_output_psnr_avg:
            state = {'model': self.model.state_dict(), 'epoch': epoch}
            self.best_output_psnr_avg,self.best_idx = PSNR_output_record.mean(),epoch+1
            torch.save(state, self.ckp_path+'/'+self.model_name+'__best')
        else:
            pass
        print('当前输出图像PSNR最大平均值：%f  来自epoch:%d'%(self.best_output_psnr_avg,self.best_idx))    
        
    # continue training
    def continue_training_isrequired(self): 
        if self.continue_training:
            print('===> Try resume from checkpoint')
            if os.path.isfile(self.pretrain_model):
                try:
                    checkpoint = torch.load(self.pretrain_model)
                    self.model.load_state_dict(checkpoint['model'])
                    ############
                    # para = model.state_dict()
                    # for i in para:
                        # print(i,para[i].size())
                    self.start_epoch = checkpoint['epoch']+1
                    print('start epoch is: ',self.start_epoch)
                    print('===> Load last checkpoint data')
                except FileNotFoundError:
                    print('Can\'t found dict')
        else: 
            self.start_epoch = 0    
            print('===> Start from scratch')  
    # test
    def tester(self, args):
        if not self.test:
            return
        if not os.path.exists(self.test_save_path):
            os.makedirs(self.test_save_path)      
        if os.path.exists(self.test_model):
            try:
                checkpoint = torch.load(self.test_model)
                self.model.load_state_dict(checkpoint['model'])
            except FileNotFoundError:
                print('Can\'t found dict')
        self.vaporTest = self.d.dataload_for_test(args)
        for batch,load in enumerate(self.vaporTest):                  
            data2, _= load
            print('test shape',data2.shape)
            if data2.shape[-2] > 1080 or data2.shape[-1] > 1428:  # if image is too large
                data2 = data2[:,:,0:1080,0:1428]
            data = torch.autograd.Variable(data2.to(self.device),requires_grad=False)
            with torch.no_grad():        
                outputTest = self.model(data)   
            print('output shape',outputTest.shape)
            outputTest = outputTest.cpu().data.numpy()*255
            h = Image.fromarray((np.squeeze(outputTest)).transpose(1,2,0).astype(np.uint8))
            h.save(self.test_save_path+self.test_name+'_'+str(batch+1)+'.png')
        print('test process is done. ')
# main 
def main():
    start = time.perf_counter()
    args = utils.args_initialize()       
    
    _model = RetroNet(args)
    tr = Train(_model, args)
    tr.trainer(args)        
    tr.tester(args)

    elapsed = (time.perf_counter() - start)
    print("Time used:",elapsed)
            
if __name__ == '__main__':
    main()            
        