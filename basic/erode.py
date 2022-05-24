import cv2
import numpy as np

#
img = cv2.imread('./dotinj.png')
# kernel=np.ones((7,7),np.uint8)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17,17))
#腐蚀
# dst = cv2.erode(img,kernel,iterations=1)
# 膨胀
# dst = cv2.dilate(img,kernel,iterations=1)

# 开运算
# dst = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

# 闭运算
# dst = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

# 梯度
# dst = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)

# 顶帽
# dst = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)

# 黑帽
dst = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)

cv2.imshow('img',img)
cv2.imshow('dst',dst)
cv2.waitKey(0)