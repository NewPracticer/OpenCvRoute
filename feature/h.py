import  cv2
import  numpy as np
# Harris
blockSize = 2
ksize = 3
k = 0.04

maxCorners =1000
q1 = 0.01
minDistance = 10

img = cv2.imread('./chess.png')
#灰度化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# dst = cv2.cornerHarris(gray,blockSize,ksize,k)
corners =cv2.goodFeaturesToTrack(gray,maxCorners,q1,minDistance)
corners = np.int0(corners)

#Harris角点展示
# img[dst>0.01*dst.max()] = [0,0,255]

#托马斯角点检测绘制
for i in corners:
  x,y = i.ravel()
  cv2.circle(img,(x,y),3,(255,0,0),-1)

cv2.imshow('Harris',img)
cv2.waitKey(0)