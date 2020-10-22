# -*- coding: utf-8 -*-
# rename images if need 
import os
for i in range(548,563):
    srcFile = 'E:\\python_exercise\\video2image\\Images\\'+str(i)+'.jpg'
    dstFile = 'E:\\python_exercise\\video2image\\Images\\'+str(i-20)+'.jpg'
    try:
        os.rename(srcFile,dstFile)
    except Exception as e:
        print(e)
        print('rename file fail\r\n')
    else:
        print('rename file success\r\n')