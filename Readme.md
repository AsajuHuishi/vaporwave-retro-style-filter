
## 简介
**[蒸汽波风格](https://www.douban.com/group/topic/120710066/?type=like)的最大特征是混合了上世纪8090年代各种标签和元素。** 蒸汽波影响最大的国家是美国，日本，中国。
美国是一个多元文化国家，任何带有混合属性的文化都很容易在活跃而包容的艺术社会中诞生和被接受。
蒸汽波的画面中出现最多的就是日文和中文。上个世纪的人们都认为，1995年的东京就是未来。在日本泡沫经济时期，人人都挥舞着万元钞票当街拦车，霓虹灯的映照下，无处不充斥着粉色和紫色的光线，人们就活在这奢靡而满足的氛围里。层出不穷的新一代电子产品，不断进步的科技生活，美丽而不切实际的未来似乎就近在眼前。
而中国作为神秘的东方文化的发源地，各种文字和东方元素都被很容易地融入到艺术作品中，充满了东方元素的科幻，本身就是一种最吸引人的流行和时尚。在后来的蒸汽波流行期，中国只是作为一个素材库出现，蒸汽波并没有很快地在国内获得很广泛的关注，近年来才突然出现在人们的视野里。而蒸汽波复古风指恢复上世纪8090年代元素的风貌、风格或风潮。**本文基于Pytorch和卷积神经网络，利用现有的主流模型方法，尝试实现图像的蒸汽波复古风格滤镜。** 本文的CSDN博客为[Python 基于卷积神经网络实现蒸汽波复古风格滤镜](https://blog.csdn.net/qq_36937684/article/details/109230671)。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201022215051651.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201022215108460.png#pic_center)

#### 1.数据获取
本文的数据集来自B站上的视频[【maru】韩国妹子的京都旅行(https://www.bilibili.com/video/BV1FE411D7JU)](https://www.bilibili.com/video/BV1FE411D7JU)和[京 都 蒸 気 少 女(https://www.bilibili.com/video/BV1o7411E7NR)](https://www.bilibili.com/video/BV1o7411E7NR)。

> BGM: Night Tempo - 夢の続き~Dreams Of Light~
采样: 【歌手】1986オメガトライブ-【歌名】Sky Surfer
原版视频:
https://www.youtube.com/watch?v=OTMKUAJLL2k
av70084254
Youtuber: maru 마루
原油管视频标注“知识共享署名许可(允许再利用)”

#### 2.数据处理 
下载视频后，首先将两个video转成图像，保存到两个文件夹（`video2image.py`）。
由于后者的视频在前者的基础上剪辑，和原版视频的顺序不太一样，所以需要对图像进行整理，使得两个文件夹下的图像内容大致对应，需要删除没有对应的图像，以及改变一些图像的序号（`rename.py`）。
另一个问题是两个文件夹的尺寸不一致，后面文件夹里的图像需要进行裁剪（`cutImages.m`），以使得图像的尺寸相同。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201022212612922.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTM3Njg0,size_16,color_FFFFFF,t_70#pic_center)

做好图像之间的对应之后它们就可以作为训练集了。我们可以选择将一些图像作为验证集（`allocateTrainValid.py`）。
本文将两个文件夹的数据使用DataLoader加载，然后再将对应的图像进行切块，确保对应图像切块的位置是相同的。

#### 3.模型选择与训练
本文选择RCAN（[Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://link.springer.com/chapter/10.1007/978-3-030-01234-2_18)）作为模型。RCAN的[Pytorch代码](https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py)非常的简便，直接拿过来用就可以了，SR的倍率设为1。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020102221231930.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTM3Njg0,size_16,color_FFFFFF,t_70#pic_center)
由于这个网络是做单张图像超分辨率（SISR）的（只是被我拉过来做复古滤镜），其他参数的设置按照默认即可，为了减小显存可以减小`n_resblock`的值，其他影响不是很大，只要loss不会异常即可。
#### 4.滤镜效果

 - 训练集中的效果（上输入，下输出）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201022214312121.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTM3Njg0,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201022214532956.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTM3Njg0,size_16,color_FFFFFF,t_70#pic_center)

结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201022214405613.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTM3Njg0,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020102221445668.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTM3Njg0,size_16,color_FFFFFF,t_70#pic_center)

 - 测试效果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201022214125648.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTM3Njg0,size_16,color_FFFFFF,t_70#pic_center)

<div align="center">
 <img src="https://img-blog.csdnimg.cn/20201022215742858.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTM3Njg0,size_16,color_FFFFFF,t_70#pic_center" >
 </div>

本文对于尺寸过大的图像会首先进行裁剪，再进行测试。

## 使用
* python<br>
* pytorch<br>
* matlab<br>


 Folder/File Name | TODO  
--------| -----------
`main.py` | 主函数（DataLoader&模型调用&训练&测试）
 model | 主流CNN模型 
 yourImages | 测试输入图像 
Data|数据集获取&处理

./Data：
Code     | TODO
-------- | -----
 `video2image.py`  | 将视频转为图像
`rename.py`   | 改变图像的序号
`cutImages.m` | 裁剪图像
`allocateTrainValid.py`|将图像分为训练集和验证集

```bash
mkdir ./log
mkdir ./test
```

训练

```bash
nohup python -u main.py -gpu -train -model_name 'RCAN' -save_model_name 'RCAN' -n_resblock 8 -bt 8 > log/retroRCAN.log &
```

测试

```bash
nohup python -u main.py -gpu -test -test_save_path './test/output/' -n_resblock 8 -test_model './checkpoint/RCAN/RCAN__best' > log/RCAN_test.log &
```





