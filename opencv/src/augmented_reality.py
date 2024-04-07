import cv2 as cv
import sys
 
img = cv.imread("/root/opencv/data/202211w15_113340.jpg")
 
if img is None:
 sys.exit("Could not read the image.")
 

scale_percent = 20
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img=resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)


cv.imshow("Display window", img)

cv.waitKey(0)
 
cv.destroyAllWindows()