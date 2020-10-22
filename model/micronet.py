# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

#######代码精简：三式改五200329
def make_model(args, parent=False):
    return Micronet5(args)

def default_conv(in_channels,out_channels,kernel_size,bias=True):
    return nn.Conv2d(in_channels,out_channels,
                     kernel_size,padding=(kernel_size//2), 
                     bias=bias)
    
def konkate(X):
    m = X[0]
#    print(m.is_cuda)
    for k in range(len(X)-1):
        tmp = X[k+1]
        m = torch.cat((tmp,m),1)
      
    return m


class Prior_Solver(nn.Module):
    def __init__(self, conv=default_conv, n_feats=32):
        super(Prior_Solver, self).__init__()
        
        kernel_size = 3
        self.convA = conv(1,n_feats,kernel_size)
        self.convB = conv(n_feats,n_feats,kernel_size)
        self.convC = conv(n_feats,n_feats,kernel_size)
        self.convD = conv(n_feats,n_feats,kernel_size)
        self.convE = conv(n_feats,n_feats,kernel_size)
        self.convF = conv(n_feats,1,kernel_size)    
        self.relu = nn.ReLU(inplace=True)
        prior_solver = [self.convA,
                        self.convB, self.relu, self.convC,
                        self.convD, self.relu, self.convE,
                        self.convF]
        self.prior_solver = nn.Sequential(*prior_solver)
        
    def forward(self,x):
        output = x + self.prior_solver(x)
        return output
        

# class Res_Block(nn.Module):
    # def __init__(self, conv=default_conv, n_feats=64):
        # super(Res_Block, self).__init__()
        
        # kernel_size = 5
        # self.conva = conv(1,n_feats,kernel_size)
        # self.convb = conv(n_feats,n_feats,kernel_size)
        # self.convc = conv(n_feats,n_feats,kernel_size)
        # self.convd = conv(n_feats,n_feats,kernel_size)
        # self.conve = conv(n_feats,n_feats,kernel_size)
        # self.convf = conv(n_feats,n_feats,kernel_size)
        # self.convg = conv(n_feats,n_feats,kernel_size)
        # self.convh= conv(n_feats,1,kernel_size)
        # self.relu = nn.ReLU(inplace=True)
    
    # def forward(self,x):
        # input1 = x
        # x = self.conva(x)
        # x = self.relu(self.convb(x))        
        # x = self.relu(self.convc(x)) 
        # x = self.relu(self.convd(x)) 
        # x = self.relu(self.conve(x)) 
        # x = self.relu(self.convf(x)) 
        # x = self.relu(self.convg(x)) 
        # x = self.convh(x)
        # output = input1 + x
        # return output        
        
## network definition
class Micronet5(nn.Module):
    def __init__(self, args, conv=default_conv, fuse=konkate):
        super(Micronet5, self).__init__()
        # self.m = m
        self.hr = args.hr
        self.lr = args.lr
#        self.batch_size = batch_size  #1
        self.left_ATA = args.left_ATA   
        self.layers = args.layers
        self.device = args.device
            
        self.left_ATA = self.left_ATA.to(self.device)
        n_feats = 32
    
        #learning parameters                        
        self.relu = nn.ReLU(inplace=True)
        
        ##1. prior solver
        modules_PS = nn.ModuleList()
        for i in range(self.layers):
            modules_PS.append(
                    Prior_Solver(n_feats=32))
            
        self.PS = nn.Sequential(*modules_PS)
        
        ##2. refinement module
        # modules_RB = nn.ModuleList()
        # for i in range(self.layers):
            # modules_RB.append(
                # Res_Block(n_feats=64))   
            
        # self.RB = nn.Sequential(*modules_RB)  
        
        ##3. tail 
        self.fuse = fuse
        modules_tail = [ #Bottleneck+conv+relu+conv
                nn.Conv2d(self.layers,n_feats,1,padding=0,stride=1),
                conv(n_feats, n_feats, kernel_size=3),
                self.relu,
                conv(n_feats, 1, kernel_size=3)]
        
        self.tail = nn.Sequential(*modules_tail)

        ##4. LADMM 
        self.lamda = nn.ParameterList()
        self.alpha = nn.ParameterList()
        self.rho1 = nn.ParameterList()
        self.rho2 = nn.ParameterList()
        self.eta = nn.ParameterList()        

        for k in range(self.layers):
            self.rho1.append(nn.Parameter(torch.ones(1, 1, dtype=torch.float32))) #四维方阵
            self.rho2.append(nn.Parameter(torch.ones(1, 1, dtype=torch.float32)))
#            self.rho2.append(nn.Parameter(torch.ones(self.m, self.batch_size, dtype=torch.float32)))
            self.lamda.append(nn.Parameter(torch.ones(1, 1, dtype=torch.float32)))
            self.alpha.append(nn.Parameter(torch.ones(1, 1, dtype=torch.float32)))
            self.eta.append(nn.Parameter(torch.ones(1, 1, dtype=torch.float32)))
    
    def forward(self, ATy):
        batch_size = ATy.shape[0]

        X0 = torch.zeros(batch_size, 1, self.hr, self.hr, dtype=torch.float32).to(self.device) #(batch_size,1,415,415)网络进入:试作甲型网络
        Z0 = torch.zeros(batch_size, 1, self.hr, self.hr, dtype=torch.float32).to(self.device) #
        M0 = torch.zeros(batch_size, 1, self.hr, self.hr, dtype=torch.float32).to(self.device) #
        mu0 = torch.Tensor([0.005]).to(self.device)
        #######################
        X = list()
        Temp = list()
        Z = list()
        M = list()
        mu = list()

        for k in range(2):
            if k == 0:
                Temp.append(X0 - ATy)
                X.append((Temp[-1]-self.rho1[k].mul(X0)-M0-mu0.mul(Z0))/(-self.rho1[k]-mu0))
                Z_a2 = Z0        
                Z_a8 = self.PS[k](Z_a2)
                Z.append((self.lamda[k]*(Z_a8)-self.rho2[k].mul(Z0)+M0-mu0.mul(X0))/(-self.rho2[k]-mu0))
                M.append(M0 + Z[-1] - X[-1])
                mu.append(self.alpha[k]*mu0)
                # X_a2 = X[-1]
                # X_a0 = self.RB[k](X_a2)
                # X[-1] = X_a0
                
            else:
                Temp.append(X[-1] - ATy)
        #        Temp.append(self.left_ATA.mul(X[-1]))#试作甲型
                X.append((Temp[-1]-self.rho1[k].mul(X[-1])-M[-1]-mu[-1].mul(Z[-1]))/(-self.rho1[k]-mu[-1]))
                Z_a2 = Z[-1]
                Z_a8 = self.PS[k](Z_a2)
                Z.append((self.lamda[k]*(Z_a8)-self.rho2[k].mul(Z[-1])+M[-1]-mu[-1].mul(X[-1]))/(-self.rho2[k]-mu[-1]))
                M.append(M[-1] + Z[-1] - X[-1])
                mu.append(self.alpha[k]*(mu[-1]))
                # X_a2 = X[-1]
                # X_a0 = self.RB[k](X_a2)
                # X[-1] = X_a0   
                
        for k in range(2,self.layers):
            Temp.append(self.left_ATA.mul(X[-1]) - self.left_ATA.mul(ATy))
    #        Temp.append(self.left_ATA.mul(X[-1]))#试作甲型
            X.append((Temp[-1]-self.rho1[k].mul(X[-1])-M[-1]-mu[-1].mul(Z[-1]))/(-self.rho1[k]-mu[-1]))
            Z_a2 = Z[-1]
            Z_a8 = self.PS[k](Z_a2)
            Z.append((self.lamda[k]*(Z_a8)-self.rho2[k].mul(Z[-1])+M[-1]-mu[-1].mul(X[-1]))/(-self.rho2[k]-mu[-1]))
            M.append(M[-1] + Z[-1] - X[-1])
            mu.append(self.alpha[k]*(mu[-1]))
            # X_a2 = X[-1]
            # X_a0 = self.RB[k](X_a2)
            # X[-1] = X_a0   
         
        output = self.tail(self.fuse(X))
        return output

    def getname(self):
        return "micronet5_naga"


  




