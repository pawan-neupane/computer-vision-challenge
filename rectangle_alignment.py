
import cv2
import numpy as np
import math
from scipy.ndimage import rotate as rotate_image
from PIL import Image, ImageDraw


import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np


angle_list = []
 
def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
  ## [visualization1]
 
def getOrientation(pts, img):

  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
 
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
  ## [visualization]
  # Draw the principal components
  cv.circle(img, cntr, 3, (255, 0, 255), 2)
  p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
  p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
  drawAxis(img, cntr, p1, (255, 255, 0), 1)
  drawAxis(img, cntr, p2, (0, 0, 255), 5)
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians

  angle_list.append(str(-int(np.rad2deg(angle)) ))
  label = "  Rotation Angle: " + str(int(np.rad2deg(angle)) - 90) + " degrees"
  textbox = cv.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
  cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
 
  return str(int(np.rad2deg(angle)) )
 

def get_angle(img):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    
    angle_list = []
    for i, c in enumerate(contours):
        area = cv.contourArea(c)
        if area < 3700 or 100000 < area:
            continue
        cv.drawContours(img, contours, i, (0, 0, 255), 2)
        a = getOrientation(c, img)
#         print(a)
        angle_list.append(a)
    return(angle_list[0])


img = cv2.imread("shapes.jpg")
TL = img[0:120,0:200]

TR = img[0:120,200:]
BL = img[120:,0:200]
BR = img[120:,200:]


a,b,c,d = get_angle(TL),get_angle(TR),get_angle(BL),get_angle(BR)



def rotate_img(img,slp):
    rotated_img1 = rotate_image(img,slp)
    cv2.imwrite("allignedshapes.jpg",rotated_img1)
    
    img = Image.open(R"allignedshapes.jpg")
    img1 = img.convert("RGB")
    seed = (5, 5)
    rep_value = (255, 255, 255)
    ImageDraw.floodfill(img, seed, rep_value, thresh=50)
    
    return(img)
  
img = cv2.imread("shapes.jpg")
print(img.shape)
TL = img[0:120,0:200]

TR = img[0:120,200:]
BL = img[120:,0:200]
BR = img[120:,200:]

TL = rotate_img(TL,int(a))
TR = rotate_img(TR,int(b))

TL = np.array(TL)
TL = cv2.resize(TL, (200,120))
TR = np.array(TR)
TR = cv2.resize(TR, (159,120))
new1 = cv2.hconcat([TL,TR])


BL = rotate_img(BL,int(c))
BR = rotate_img(BR,int(d))

BL = np.array(BL)
BL = cv2.resize(BL, (200,326))
BR = np.array(BR)
BR = cv2.resize(BR, (159,326))
new2 = cv2.hconcat([BL,BR])

new3 = cv2.vconcat([new1,new2])

cv2.imwrite("allignedshapes.jpg",new3)

cv2.imshow('image ', new3)
cv2.waitKey(0)
cv2.destroyAllWindows



