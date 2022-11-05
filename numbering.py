import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def min_dist(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,5,3,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    corners = corners[1:]
    corners = sorted(corners, key=lambda x: x[0])

    corners_left = corners[0:3]
    corners_left = sorted(corners_left, key=lambda x: x[1])
    corners_right = corners[3:]
    corners_right = sorted(corners_right, key=lambda x: x[1])
    img[dst>0.1*dst.max()]=[0,0,255]

    distances = []
    for i in range(0,2):
        x1,y1 = corners_left[i][0],corners_left[i][1]
        x2,y2 = corners_right[i][0],corners_right[i][1]
        distance = round(math.sqrt( ((x1-x2)**2)+((y1-y2)**2) ),ndigits= 5) 
        distances.append(distance)
    
    print(distances)
    distances.sort()  
    return(distances[0])



def print_name(img,name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h,w,c = img.shape
    org = (int(w/2)-20, h-20)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    name = name
    image = cv2.putText(img, name, org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
    return(image)



img = cv2.imread("shapes.jpg")

TL = img[0:120,0:200]
TR = img[0:120,200:]
BL = img[120:,0:200]
BR = img[120:,200:]

# d = min_slope(lefttop)
d1,d2,d3,d4 = min_dist(TL),min_dist(TR),min_dist(BL),min_dist(BR)
distance_list = [d1,d2,d3,d4]

distance_list.sort()
TL_string = str(distance_list.index(d1)+1)+"(p:"+str(int(d1))+")"
TR_string = str(distance_list.index(d2)+1)+"(p:"+str(int(d2))+")"
BL_string = str(distance_list.index(d3)+1)+"(p:"+str(int(d3))+")"
BR_string = str(distance_list.index(d4)+1)+"(p:"+str(int(d4))+")"

TL = print_name(TL, TL_string)
# cv2.imwrite("TL.jpg",TL)
TR = print_name(TR, TR_string)
img_final1 = cv2.hconcat([TL, TR])
# cv2.imwrite("TR.jpg",TR)
BL = print_name(BL, BL_string)
# cv2.imwrite("BL.jpg",BL)
BR = print_name(BR, BR_string)
img_final2 = cv2.hconcat([BL, BR])

img_final = cv2.vconcat([img_final1, img_final2])

cv2.imwrite("numberedshapes.jpg",img_final)

cv2.imshow('image p = length in pixel', img_final)
cv2.waitKey(0)
cv2.destroyAllWindows