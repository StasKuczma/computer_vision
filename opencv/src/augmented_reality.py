import cv2 
import sys
#import image
img = cv2.imread("/root/opencv/data/20221115_113319.jpg")

#no image handle
if img is None:
 sys.exit("Could not read the image.")
 
#resize image
scale_percent = 20
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

#convert image to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect markers
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

corners, ids, rejected = detector.detectMarkers(gray)

#draw markers
cv2.aruco.drawDetectedMarkers(img, corners, ids)   
cv2.imshow("Display window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()