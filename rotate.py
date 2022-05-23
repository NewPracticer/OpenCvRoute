import cv2
import numpy as np

desk = cv2.imread('./desk.jpg')
new = cv2.rotate(desk,cv2.ROTATE_90_CLOCKWISE)

cv2.imshow('rotate',new)
cv2.waitKey(0)