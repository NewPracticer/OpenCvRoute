import cv2
import  numpy as np

img = cv2.imread('./chess.png')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sift = cv2.xfeatures2d.SIFT_create()

# surf = cv2.xfeatures2d.SURF_create()

orb = cv2.ORB_create();

kp,des = orb.detectAndCompute(gray,None)

# print(des)
cv2.drawKeypoints(gray,kp,img)

cv2.imshow('img',img)
cv2.waitKey(0)