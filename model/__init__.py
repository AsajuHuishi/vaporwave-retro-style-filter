import os
from importlib import import_module
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel as P

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
import random
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.name = args.model_name
        print('Making model...')
  
        self.module = import_module('model.' + args.model_name.lower()) # 动态导入对象
        self.model = self.module.make_model(args)
        print('Making model '+str(args.model_name)+' is done.')

    def forward(self,x):
        return self.model(x)
        
    def name(self):
        return self.module.getname()

        