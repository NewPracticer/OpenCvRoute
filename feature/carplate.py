import cv2
import numpy as np

# 引入tessact库
import  pytesseract

# 第一步 创建Haar级联器
faser = cv2.CascadeClassifier('./haarcascades/haarcascade_russian_plate_number.xml')

#第二步，带车牌的图片
img = cv2.imread('./chinacar.jpeg')

#第三步 车牌定位
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#第三步 检测车牌位置
# [[x,y,w,h]]
plate = faser.detectMultiScale(gray,1.1,3)
for(x,y,w,h) in plate:
  cv2.rectangle(img, (x,y), (x+w,y+h),(0,0,255),2)

# 对获取到的车牌进行预处理
# 1.提取ROI
roi = gray[y:y+h, x:x+w]
# 2. 进行二值化
ret, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

text = pytesseract.image_to_string(roi_bin, lang='chi_sim+eng',config= '--psm 8 --oem 3')

print(text)

cv2.imshow('roi_bin',roi_bin)
cv2.imshow('img',img)
cv2.waitKey(0)





