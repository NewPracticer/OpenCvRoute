import  cv2
import numpy as np

coffee = cv2.imread('./coffee.jpg')
coffee2 = cv2.imread('./coffee2.jpg')

result = cv2.addWeighted(coffee,0.7,coffee2,0.3,0)

desk = cv2.imread('./desk.jpg')
desk[50:150,50:150] = 0
result2 = cv2.bitwise_not(desk)
result3 = cv2.bitwise_and(desk,result2)
result4 = cv2.bitwise_or(desk,result2)
result5 = cv2.bitwise_xor(desk,result2)

cv2.imshow('add2',result)
cv2.imshow('yihuo',result2)
cv2.imshow('and',result3)
cv2.imshow('or',result4)
cv2.imshow('xor',result5)
cv2.waitKey(0)



