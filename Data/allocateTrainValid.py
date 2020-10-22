'''
@author: AsajuHuishi
@Date: 20/10/22
'''

import os
validList = [12, 478, 512, 543]

os.system("rm -rf ./Data_train/Images/*")
os.system("rm -rf ./Label_train/Images/*")
os.system("rm -rf ./Data_valid/Images/*")
os.system("rm -rf ./Label_valid/Images/*")
for i in validList:
    os.system(" cp ./ImagesCut/%d.jpg ./Data_valid/Images/ "%(i))
    os.system(" cp ./ImagesGT/Images/%d.jpg ./Label_valid/Images/ "%(i))

os.system("cp ./ImagesCut/* ./Data_train/Images/")
os.system("cp ./ImagesGT/Images/* ./Label_train/Images/")