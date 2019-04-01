#Shreyas Narasimha
#UBID: sn58
#UB Person No. : 50289736

import cv2
import numpy as np
import math
import time
s = time.time()

def display(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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

def opening(img):
    a = dilate(erode(img))
    return a

def closing(img):
    a = erode(dilate(img))
    return a

    
def edge_detect(img):
    h,w = img.shape
    edge_x = np.zeros((img.shape))
    edge_y = np.zeros((img.shape))
    kernel1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    kernel2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
           edge_x[i,j] = np.sum(kernel1*img[i-1:i+2,j-1:j+2])
           edge_y[i,j] = np.sum(kernel2*img[i-1:i+2,j-1:j+2])
    
    a = np.sqrt(np.square(edge_x)+np.square(edge_y))
    a = np.asarray(a,dtype=np.uint8)
    
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if(a[i,j] > 100 ):
                a[i,j] = 255
            else:
                a[i,j] = 0
    #a = cv2.threshold(a,100,200,cv2.THRESH_BINARY)[1]
    return a         

def med(p1):
    temp1 = []
    p = p1.tolist()
    p.append([-999,-999])
    mean_theta = np.mean(p1,axis = 0)
    i = 0
    while(i < p1.shape[0]):
        summ = 0
        count = 0
        while(abs(p[i][0] - p[i+1][0]) < 12):
            summ = summ + p[i][0]
            count = count + 1
            i+=1            
        summ = summ + p[i][0]
        i+=1
        mean = summ/(count+1)
        temp1.append([mean,mean_theta[1]])
    return temp1


def hough(image,img):    
    h,w = image.shape
    temp = image.copy()
    img1 = img.copy()
    img2 = img.copy()
    points = []
    #temp = cv2.Canny(img,100,200)
    temp = cv2.GaussianBlur(image,(3,3),5)
    temp = edge_detect(temp)
    
    temp = erode(opening(closing(temp)))
    angle = 90
    a = np.zeros((int(2*np.sqrt(np.square(h)+np.square(w))),(2*angle)+1))
    for i in range(h):
        for j in range(w):
            #print(temp.shape)
            if(temp[i,j] > 0):
                for k in range(-angle,angle+1):
                    r = (j*np.cos(np.radians(k)) + i*np.sin(np.radians(k)))+ np.sqrt(np.square(h)+np.square(w))
                    a[int(r),k+angle]+=1
                    
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if(a[i,j] > 130):
                points.append([i,j])
            
    vpoints = []
    spoints = []
    for i in points:
         if( -2 <= i[1]-90 <= -2):
             vpoints.append(i)
         elif((-37 <= i[1]-90 <= -36)):
                spoints.append(i)
                
    v2 = med(np.asarray(vpoints))
    v1 = med(np.asarray(spoints))
    for i in v2:
        a1 = np.cos(np.radians(i[1]-90))
        b1 = np.sin(np.radians(i[1]-90))
        x0 = a1*(i[0] - np.sqrt(np.square(h)+np.square(w)))
        y0 = b1*(i[0] - np.sqrt(np.square(h)+np.square(w)))
        x1 = int(x0 + 1000*(-b1))
        y1 = int(y0 + 1000*(a1))
        x2 = int(x0 - 1000*(-b1))
        y2 = int(y0 - 1000*(a1))
        cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)
    
    for i in v1:
        a1 = np.cos(np.radians(i[1]-90))
        b1 = np.sin(np.radians(i[1]-90))
        x0 = a1*(i[0] - np.sqrt(np.square(h)+np.square(w)))
        y0 = b1*(i[0] - np.sqrt(np.square(h)+np.square(w)))
        x1 = int(x0 + 1000*(-b1))
        y1 = int(y0 + 1000*(a1))
        x2 = int(x0 - 1000*(-b1))
        y2 = int(y0 - 1000*(a1))
        cv2.line(img1,(x1,y1),(x2,y2),(255,0,0),2)   
    
    return img1,img2
                
def circon(p1):
    temp1 = []
    p = p1.tolist()
    p.append([-999,-999,-999])
    mean_r = np.mean(p1,axis = 0)[2]
    #p = sort_points(p)
    i = 0
    while(i < p1.shape[0]):
        summ1 = 0
        count1 = 0
        summ2 = 0
        count2 = 0
        while((np.square(p[i][0] - p[i+1][0])) + (np.square(p[i][1] - p[i+1][1])) < 9):
            summ1 = summ1 + p[i][0]
            count1 = count1 + 1
            summ2 = summ2 + p[i][0]
            count2 = count2 + 1
            i+=1            
        summ1 = summ1 + p[i][0]
        summ2 = summ2 + p[i][1]
        i+=1
        mean1 = summ1/(count1+1)
        mean2 = summ2/(count2+1)
        temp1.append([mean1,mean2,mean_r])
    return temp1
    
def unique(p):
     a = p.copy()
     for i in range(len(a)):
         a[i] = tuple(a[i])
     a = tuple(tuple(a))
     a = set(a)
     a = list(a)
     return a
     
def chough(image,img):
    h,w = image.shape
    temp = image.copy()
    img1 = img.copy()
    #temp = cv2.GaussianBlur(image,(3,3),2)
    temp = edge_detect(temp)
    #temp = closing(temp)
    temp = dilate(temp)
    angle = 360
    diagonal = np.sqrt(np.square(h)+np.square(w))
    acc = np.zeros((w,h,int(diagonal)))
    #r = 24
    for i in range(h):
        for j in range(w):
            if(temp[i,j] > 200):
                for r in range(23,25):
                    for k in range(angle):
                        a = j - r*np.cos(np.deg2rad(k))
                        b = i - r*np.sin(np.deg2rad(k))
                        if((a>=(w-1)) or (b>=(h-1))):
                            continue
                        else:
                            acc[int(a),int(b),r]+=1
    print(acc)
    points = []                
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            for k in range(acc.shape[2]):
                if(acc[i,j,k] > 330):
                    points.append([i,j,k])
                    #print(acc[i,j,k])
    #points = circon(np.asarray(points))
    points = unique(points)
    for i in points:    
        cv2.circle(img1,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),thickness = 2)
    return img1


img = cv2.imread("original_imgs/hough.jpg", 0)
cimg = cv2.imread("original_imgs/hough.jpg")
height,width = img.shape
print('Running Hough transform for lines\n')
houghed = hough(img,cimg)
print('Running Hough transform for coins')
choughed = chough(img,cimg)

cv2.imwrite('coin.jpg',choughed)
cv2.imwrite('red_line.jpg',houghed[1])
cv2.imwrite('blue_lines.jpg',houghed[0])

print(time.time() - s)

