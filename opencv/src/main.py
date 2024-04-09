import cv2 
import sys
import numpy as np
#import image
base = cv2.imread("/root/opencv/data/20221115_113328.jpg")
# base = cv2.imread("/root/opencv/data/20221115_113319.jpg")

img = cv2.imread("/root/opencv/data/town.jpg")

#no image handle
if base is None:
 sys.exit("Could not read the image.")
 
def resize(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


#resize images to fit into the screen
# base = resize(base, 20)

#detect markers
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
corners, ids, rejected = detector.detectMarkers(base)
cv2.aruco.drawDetectedMarkers(base, corners, ids)   
corner = corners[0][0]

dim=[int(corner[0][1]),int(corner[2][1]),int(corner[0][0]),int(corner[2][0])]

#getting the corner of the image
height, width = img.shape[:2]
top_left = [0, 0]
top_right = [width/3 - 300, 0]
bottom_left = [0, height/3 - 1]
bottom_right = [width/3 -300, height/3 -1]



pts1 = np.float32([top_left,top_right,bottom_right,bottom_left])#image base
pts2=np.float32([[corner[0][0]-50,corner[0][1]-50],[corner[1][0]-50,corner[1][1]-50],[corner[2][0]-50,corner[2][1]-50],[corner[3][0]-50,corner[3][1]-50]])

print(pts1)
print(corner)



M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(base.shape[1],base.shape[0]))


rows,cols,channels = dst.shape
roi = base[0:rows, 0:cols]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
 
# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
 
# # Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(dst,dst,mask = mask)
 
# # Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
base[0:rows, 0:cols ] = dst

cv2.imshow("Display window", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()