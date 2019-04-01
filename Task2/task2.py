#Shreyas Narasimha
#UBID: sn58
#UB Person No. : 50289736

import cv2
import numpy as np
import math
import time
from matplotlib import pyplot as plt
s = time.time()

def display(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_points(image,ks):
    h = int(image.shape[0])
    w = int(image.shape[1])
    temp = image.copy()
    temp = temp*0
    for i in range(int(ks/2),h-int(ks/2)):
        for j in range(int(ks/2),w-int(ks/2)):
            inter = image[i-int(ks/2):i+int(ks/2)+1, j-int(ks/2):j+int(ks/2)+1]*kernel
            center = np.absolute(np.sum(inter))
            #temp[i][j] = center
            if(center > 2300):
                temp[i][j] = center
    return temp

def dilate(image):
    h = int(image.shape[0])
    w = int(image.shape[1])
    temp = image.copy()
    temp = temp*0
    for i in range(1,h-1):
        for j in range(1,w-1):
            if(image[i][j] > 100):
                temp[i-1:i+2,j-1:j+2] = 255
    return temp

def segment(image):
    h = int(image.shape[0])
    w = int(image.shape[1])
    t_next = 10
    t1 = 0
    dif = 999    
    temp = image.copy() *0
    while(dif > 10):
        t1 = t_next
        count1 = 0
        count2 = 0
        sum1 = 0
        sum2 = 0
        for i in range(h):
            for j in range(w):
                if(image[i][j] > t1):
                    count1 = count1+1
                    sum1+= image[i][j]                    
                else:
                    count2 = count2+1
                    sum2+= image[i][j]
        if(count1!=0):
            m1 = sum1/(1.0*count1)
        else:
            m1 = 0
        t_next = m1
        dif = t_next - t1
        #print(t_next)
    
    for i in range(h):
            for j in range(w):
                if(image[i][j] > t_next):
                    temp[i][j] = 255
    #cv2.line(image,(5,5),(45,45),(255,0,0),5)
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

def opening(img):
    a = dilate(erode(img))
    return a

def closing(img):
    a = erode(dilate(img))
    return a
#Function to find bounding boxes
def bounds(img,cimg1):
    #img = cv2.imread('segment.jpg') 
    img1 = img.copy()
    cimg = cimg1.copy()
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for i in range(4):         
        template = cv2.imread('templates/bone'+str(i)+'.png',0)
        h, w = template.shape[::-1]
        res = cv2.matchTemplate(img,template,cv2.TM_CCORR_NORMED)
        threshold = 0.99
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(cimg, pt, (pt[0] + h, pt[1] + w), (255,255,255 ), 1)
            cv2.rectangle(img1, pt, (pt[0] + h, pt[1] + w), (255,255,255 ), 1)
            print('For object ' + str(i) + '\n')
            print('The top left corner(column,row) ' + '(' + str(pt[0]),str(pt[1]) +')' )
            print('The top right corner(column,row) ' + '(' + str(pt[0]+h),str(pt[1]) +')' )
            print('The bottom left corner(column,row) ' + '(' + str(pt[0]),str(pt[1]+w) +')' )
            print('The bottom right corner(column,row) ' + '(' + str(pt[0]+h),str(pt[1]+w) +')' )
            print('\n \n')
    return cimg,img1
   
ksize = 5
kernel = np.ones((ksize, ksize))*-1
kernel[int(ksize/2)][int(ksize/2)] = (ksize*ksize) - 1
#kernel[int(ksize/2)][int(ksize/2)] = 24

img1 = cv2.imread("original_imgs/point.jpg", 0)
img2 = cv2.imread("original_imgs/segment.jpg",0)
print('Running point detection and segmentation \n')
thresh = detect_points(img1,ksize)
position = np.unravel_index(np.argmax(thresh), thresh.shape)
thresh = dilate(thresh)
cat = segment(img2)
#cat = closing(opening(cat))
cat = erode(cat)

bound1,bound2 = bounds(cat,img2)
print('The detected point is at coordinates: \n' + str(position))
cv2.imwrite('bounded.jpg',bound2)
cv2.imwrite('bounded_original.jpg',bound1)
cv2.imwrite('segment.jpg',cat)
cv2.imwrite('point.jpg',thresh)

print(time.time() - s)