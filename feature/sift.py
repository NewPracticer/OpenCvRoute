import cv2
import  numpy as np

img1 = cv2.imread('./opencv_search.png')
img2 = cv2.imread('./opencv_logo.png')

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

# surf = cv2.xfeatures2d.SURF_create()

# orb = cv2.ORB_create();

# kp,des = sift.detectAndCompute(gray,None)
kp1,des1 = sift.detectAndCompute(gray1,None)
kp2,des2 = sift.detectAndCompute(gray2,None)

bf = cv2.BFMatcher(cv2.NORM_L1)
match = bf.match(des1,des2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,match,None)

# print(des)
# cv2.drawKeypoints(gray,kp,img)

cv2.imshow('img',img3)
cv2.waitKey(0)