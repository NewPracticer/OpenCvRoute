import  cv2
import numpy as np

img = cv2.imread('./lena.png')

#索贝尔算子 y方向边缘
d1 = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
d2 = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

d3 = cv2.Scharr(img,cv2.CV_64F,0,1)

d4 = cv2.Laplacian(img,cv2.CV_64F,ksize=5)

dst = cv2.Canny(img, 100,200)

cv2.imshow('img',img)
cv2.imshow('d1',d1)
cv2.imshow('d2',d2)
cv2.imshow('d3',d3)
cv2.imshow('d4',d4)
cv2.imshow('dst',dst)
cv2.waitKey(0)