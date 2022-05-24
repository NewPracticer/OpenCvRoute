import  cv2
import numpy as np

img = cv2.imread('e:\\1394506194911.jpg')
# 高度，长度和通道数
print(img.shape)
# 占用多大空间
# 高度×长度×通道数
print(img.size)
# 图像中每个元素的位深
print(img.dtype)


