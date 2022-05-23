import  cv2
import numpy as np

coffee = cv2.imread('./coffee.jpg')
new = cv2.resize(coffee,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_CUBIC)

cv2.imshow('new',new)
cv2.waitKey(0)