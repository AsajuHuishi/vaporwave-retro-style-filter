# -*- coding: utf-8 -*-
import random
from PIL import Image
import numpy as np

def get_patch(*args, patch_size=256, scale=1, multi=False):
    ih, iw = args[0].shape[:2]
    p = scale if multi else 1
    tp = p * patch_size
    ip = tp

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret

#def main():
# 图片路径，相对路径
image_path0 = ".\\ImagesCut\\Images\\110.jpg"
image_path1 = ".\\ImagesGT\\Images\\110.jpg"
# 读取图片
image0 = Image.open(image_path0)
image1 = Image.open(image_path1)
# 显示图片
#    image.show()
img0 = np.array(image0)
img1 = np.array(image1)
print(img0.shape) #(1080, 1428, 3)
print(img1.shape) #(1080, 1428, 3)

#img0 = np.random.random((1200,1200,3)).astype(np.uint8)
#img1 = np.random.random((1200,1200,3)).astype(np.uint8)
#img0 = torch.from_numpy(img0)
#img1 = torch.from_numpy(img1)
ret = get_patch(img0,img1)
print(ret[0].shape)
c = Image.fromarray(ret[0])
c.show()
c = Image.fromarray(ret[1])
c.show()
#if __name__ == '__main__':
#    main()