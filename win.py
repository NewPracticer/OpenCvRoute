import cv2
import numpy as np
# 显示窗口
# cv2.namedWindow('new',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('new',1920,1080)
# cv2.imshow('new',0)
# key = cv2.waitKey(100)
# if(key == 'q'):
#     exit()
# cv2.destroyAllWindows()

# 显示图片
# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
img = cv2.imread('e:\\1394506194911.jpg')
# cv2.imshow('image',img)
# cv2.waitKey(100)

#保存图片
# cv2.imwrite('e:\\123.png',img)

# 读取视频流
# cv2.namedWindow('video',cv2.WINDOW_AUTOSIZE)
# # 获取视频设备
# cap = cv2.VideoCapture(0)
# while True:
#   ret,frame = cap.read()
#   cv2.imshow('video',frame)
#   # 等待键盘事件
#   key = cv2.waitKey(10)
# # 释放videoCapture
# cap.release()
# cv2.destroyAllWindows



# 读取视频文件
# cap = cv2.VideoCapture('视频所在地址')
# while cap.isOpened():
#   ret,frame = cap.read()
#   cv2.imshow('video',frame)
#   # 等待键盘事件
#   key = cv2.waitKey(10)
# # 释放videoCapture
# cap.release()
# cv2.destroyAllWindows



# 视频录制
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# vm = cv2.VideoWriter('./out.mp4',fourcc,25,(1280,720))



# 使用isOpened()判断摄像头是否已打开



# # 鼠标事件监听
# def mouse_callback(event,x,y,flags,userdata):
#     print(event,x,y,flags,userdata)
# mouse_callback(1,100,100,16,'666')
# #创建窗口
# cv2.namedWindow('mouse',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('mouse',640,360)
# 设置鼠标回调
# cv2.setMouseCallback('mouse',mouse_callback,'123')
# img = np.zeros((360,640,3),np.uint8)
# while True:
#     cv2.imshow('mouse',img)
#     key = cv2.waitKey(1)
#     if key & 0xFF == ord('q'):
#         break;
# cv2.destroyAllWindows()



# TrackBar控件
# def callable():
#     pass
# cv2.namedWindow('trackbar',cv2.WINDOW_AUTOSIZE)
#
# cv2.createTrackbar('R','trackbar',0,255,callable)
# cv2.createTrackbar('G','trackbar',0,255,callable)
# cv2.createTrackbar('B','trackbar',0,255,callable)
#
# img = np.zeros((480,460,3),np.uint8)
#
# while True:
#     r = cv2.getTrackbarPos('R','trackbar')
#     g = cv2.getTrackbarPos('G','trackbar')
#     b = cv2.getTrackbarPos('B','trackbar')
#
#     img[:] = [b,g,r]
#     cv2.imshow('trackbar', img)
#
#     cv2.waitKey(10)
#
# cv2.destroyAllWindows()

# 色彩空间转换
# def callback():
#     pass
# cv2.namedWindow('color',cv2.WINDOW_NORMAL)
# img = cv2.imread('e:\\1394506194911.jpg')
#
# colorspaces = [cv2.COLOR_BGR2RGBA,cv2.COLOR_BGR2BGRA,
#                cv2.COLOR_BGR2GRAY,cv2.COLOR_BGR2HSV_FULL,
#                cv2.COLOR_BGR2YUV]
# cv2.createTrackbar('curcolor','color',0,len(colorspaces),callback)
#
# while True:
#     index = cv2.getTrackbarPos('curcolor','color')
#
#     # 颜色空间转换API
#     cvt_img = cv2.cvtColor(img,colorspaces[index])
#     cv2.imshow('color',img)
#     cv2.waitKey(10)

# Numpy使用
# 通过array定义矩阵
a = np.array([1,2,3])
b = np.array([[1,2,3],[4,5,6]])
print(a)
print(b)
# 定义zeros矩阵
c = np.zeros((8,8,3),np.uint8)
# (480,640,3) (行的个数、列的个数，通道数/层数)
# np.uint8 矩阵中的数据类型
print(c)
c = np.zeros((8,8),np.uint8)
print(c)
# 定义Full矩阵
e = np.full((8,8),255,np.uint8)
print(e)
# 定义单位矩阵 indentity 使用
f = np.identity(4)
print(f)
# eye创建矩阵
g = np.eye(5,7)
print(g)
# 检索与赋值[y,x]
# [y,x,channel]
print(img[100,100])
count = 0
# while count<200:
#     img[count,100] =[0,0,255]
#     count = count +1
# img[100,100] = 255
# cv2.imshow('img',img)
# cv2.waitKey(0)
# 获取子矩阵 Region of Image (ROI)[y1:y2,x1:x2]
roi = img[100:400,100:600]
roi[:,:] = [0,0,255]
roi[:,10] = [0,0,0]
roi[10:200,10:200]= [0,255,0]
cv2.imshow('img',img)
cv2.waitKey(0)





