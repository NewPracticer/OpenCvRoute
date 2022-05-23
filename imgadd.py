import  cv2
import  numpy as np

# 图的加法运算 就是矩阵的加法运算
# 因此，加法运算的两张图必须是相等的

coffee = cv2.imread('./coffee.jpg')
print(coffee.shape)

img = np.ones((1024,768,3),np.uint8) * 50
cv2.imshow('org',coffee)
result = cv2.add(coffee,img)
result2 = cv2.subtract(coffee,img)
result3 = cv2.multiply(coffee,img)
result4 = cv2.divide(coffee,img)
cv2.imshow('result',result)
cv2.imshow('result2',result2)
cv2.imshow('result3',result3)
cv2.imshow('result4',result4)
cv2.waitKey(0)
