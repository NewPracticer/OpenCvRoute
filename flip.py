import cv2
import numpy as np

desk = cv2.imread('./desk.jpg')
new = cv2.flip(desk,0)
new1 = cv2.flip(desk,1)

cv2.imshow('flip1',new)
cv2.imshow('flip2',new1)
cv2.waitKey(0)