import cv2 
import sys
import numpy as np
 
#resize image just for visualization
def resize(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def detect_marker(img):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids = detector.detectMarkers(img)[:2]
    cv2.aruco.drawDetectedMarkers(img, corners, ids)   
    return corners[0][0]

def perspective_transform(img, pts1, pts2):
    M = cv2.getPerspectiveTransform(pts1,pts2)
    wraped = cv2.warpPerspective(img,M,(base.shape[1],base.shape[0]))
    return wraped

def paste_image(transformed_image,base):
    height,width = transformed_image.shape[:2]
    roi = base[0:height, 0:width]
    #creating a mask and inverting it
    img2gray = cv2.cvtColor(transformed_image,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    #Black-out the area of painting in ROI
    img1_bg = cv2.bitwise_and(base,base,mask = mask_inv)
    cv2.imwrite("/root/opencv/data/painting/img_bg"+str(i+1)+".jpg", img1_bg)
    #combining the painting and the ROI
    base[0:height, 0:width ] = cv2.add(img1_bg,transformed_image)
    return base
def show_image(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("/root/opencv/data/town.jpg")

for i in range(11):
    base = cv2.imread("/root/opencv/data/"+str(i+1)+".jpg")

    if base is None:
        sys.exit("Could not read the image.")

    try:
        #detecting corners of aruco marker
        corner = detect_marker(base)

        #casting corners of image to aruco marker
        cast1 = np.float32([[0, 0],
                            [img.shape[1]/3 - 300, 0],
                            [img.shape[1]/3 -300, img.shape[0]/3 -1],
                            [0, img.shape[0]/3 - 1]])
        


        cast2 = np.float32([[corner[0][0],corner[0][1]],
                            [corner[1][0],corner[1][1]],
                            [corner[2][0],corner[2][1]],
                            [corner[3][0],corner[3][1]]])
        

        transformed_image=perspective_transform(img, cast1, cast2)
        ready_image = paste_image(transformed_image, base)

        # cv2.imwrite("/root/opencv/data/painting/"+str(i+1)+".jpg", ready_image)
        cv2.imwrite("/root/opencv/data/painting/test"+str(i+1)+".jpg", ready_image)

        print("Image", i+1, "has been processed.")
    except:
    
        print("Image", i+1, "does not have an Aruco marker.")
        cv2.imwrite("/root/opencv/data/painting/"+str(i+1)+".jpg", base)



