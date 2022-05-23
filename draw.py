import  cv2
import  numpy as np

img = np.zeros((480,640,3),np.uint8)
#画线，坐标点为（x,y）
cv2.line(img,(10,20),(300,400),(0,0,255),5,4)

cv2.line(img,(10,20),(200,300),(0,0,255),5,16)

# 画椭圆
# 度数是按照顺时针计算的
# 零度是从右侧开始
cv2.ellipse(img,(320,240),(100,50),45,0,360,(0,0,255),-1)

#绘制多边形
pts = np.array([(300,10),(150,150),(150,100),(450,100)],np.int32)
cv2.polylines(img,[pts],True,(0,0,255))
# 填充多边形
cv2.fillPoly(img,[pts],(255,0,0))

# 绘制文本
cv2.putText(img,'Hello world',(10,400),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))

cv2.imshow('draw',img)
cv2.waitKey(0)

