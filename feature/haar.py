import cv2
import numpy as np

# 第一步 创建Haar级联器
faser = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

eye = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
# 第二步 导入人脸检测图片，并将其灰度化

img = cv2.imread('./p3.png')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#第三步　进行人脸识别　
# [[x,y,w,h]]
faces = faser.detectMultiScale(gray,1.1,5)

i = 0
for(x,y,w,h) in faces:
  cv2.rectangle(img, (x,y), (x+w,y+h),(0,0,255),2)
  roimg = img[y:y+h, x:x+w]
  eyes = eye.detectMultiScale(roimg,1.1,3)
  for (x,y,w,h) in eyes:
      cv2.rectangle(roimg, (x,y), (x+w,y+h), (0,255,0), 2)
  i = i+1
  winname = 'face'+str(i)
  cv2.imshow(winname, roimg)

cv2.imshow('img',img)
cv2.waitKey(0)


