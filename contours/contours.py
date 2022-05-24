import  cv2
import  numpy as np

img = cv2.imread('./contours1.jpeg')
# 转变成单通道
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#二值化
ret, binary = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)

countours,hiearchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
cv2.drawContours(img,countours,-1,(0,255,0),1)

# 轮廓的面积和周长
area = cv2.contourArea(countours[0])
print('area=%d'%(area))

# 计算周长
len = cv2.arcLength(countours[0],True)

print('len=%d'%(len))

cv2.imshow('img',img)
cv2.imshow('bin',binary)
cv2.waitKey(0)
