#Shreyas Narasimha
#UBID: sn58
#UB Person No. : 50289736

import cv2
import numpy as np
import math
import time
s = time.time()
def dilate(image):
    h = int(image.shape[0])
    w = int(image.shape[1])
    temp = image.copy()
    temp = temp*0
    for i in range(1,h-1):
        for j in range(1,w-1):
            if(image[i][j] == 255):
                temp[i-1:i+2,j-1:j+2] = 255
    return temp

def erode(image):
    kernel = np.ones((3,3))
    kernel = kernel*255
    h = int(image.shape[0])
    w = int(image.shape[1])
    temp = image.copy()
    temp = temp*0
    for i in range(1,h-1):
        for j in range(1,w-1):
            if(np.array_equal(kernel,image[i-1:i+2,j-1:j+2])):
                temp[i][j] = 255
            else:
                temp[i][j] = 0
    return temp       
#display function
def display(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def opening(img):
    a = dilate(erode(img))
    return a

def closing(img):
    a = erode(dilate(img))
    return a

def boundary(img):
    a = erode(img)
    a = img - a
    return a 
#the 3x3 kernel
img = cv2.imread("original_imgs/noise.jpg", 0)
(h,w) = img.shape

#Converting image to binary
thresh = 127
bin_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
#Boundary
o = opening(closing(img))
c = closing(opening(img))
b1 = boundary(o)
b2 = boundary(c)
#Opening and Closing
cv2.imwrite('res_noise1.jpg',o)
cv2.imwrite('res_noise2.jpg',c)
cv2.imwrite('res_bound1.jpg',b1)
cv2.imwrite('res_bound2.jpg',b2)

print(time.time() - s)