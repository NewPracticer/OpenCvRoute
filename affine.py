import cv2
import numpy as np

desk = cv2.imread('./desk.jpg')
h,w,ch = desk.shape
# M = np.float32([[1,0,100],[0,1,0]])
# 旋转的角度为逆时针
# M = cv2.getRotationMatrix2D((100,100),15,0.3)

src = np.float32([[400,300],[800,300],[400,1000]])
dst = np.float32([[200,400],[600,500],[150,1100]])
M = cv2.getAffineTransform(src,dst)
new = cv2.warpAffine(desk,M,(w,h))
cv2.imshow('new',new)
cv2.waitKey(0)