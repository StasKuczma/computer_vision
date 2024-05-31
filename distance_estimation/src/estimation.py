import cv2


path="/root/yolov9/data_kitti/kitti_yolo/img/006374.png"

label_path="/root/yolov9/data_kitti/kitti_yolo/labels/006374.txt"

K_path="/root/yolov9/data_kitti/kitti_gt/calib/006374.txt"

# Load an image using 'imread' specifying the path to image
image = cv2.imread(path)

# Display the image in a window named 'image'


with open(label_path, 'r') as file:
    # Read the content of the file
    label = file.read()
    label = label.split("\n")
    label = [x.split(" ") for x in label]
    
    label = [[int(y) for y in x] for x in label[:2]]

with open(K_path, 'r') as file:
    # Read the content of the file
    K = file.read()
    K = K.split("\n")
    K = [x.split(" ") for x in K]
    K = [[float(y) for y in x] for x in K[:3]]


cv2.imshow('image', image)



# Wait for any key to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()