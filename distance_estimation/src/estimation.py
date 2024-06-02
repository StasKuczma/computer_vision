# import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Define the camera height
camera_height=1.65

# Define the lists for the distances
distance_estimated=[]
distance_gt=[]

# Define the directory
directory = '/root/yolov9/data_kitti/kitti_gt/labels'

# Loop through all files in the directory
for filename in os.listdir(directory):
    empty_file = False

    # Read the K matrix
    K_path="/root/yolov9/data_kitti/kitti_gt/calib/"+filename
    with open(K_path, 'r') as file:
        K = file.read()
        if K == '':
            empty_file = True
            print('Empty file')
        else:
            #Split file into lines and then split lines into elements
            K=[K.split(" ") for K in K.split("\n")]
            #Exclude last element which is empty
            K = [[float(y) for y in x] for x in K[:3]]
            #Convert to numpy array
            matrix = np.array(K)

    # Read the label form yolo
    label_path_yolo="/root/yolov9/data_kitti/kitti_yolo/labels/"+filename
    with open(label_path_yolo, 'r') as file:
        label = file.read()
        #If not empty calculate
        if label == '':
            empty_file = True
            print('Empty file')
        else:    
            #Split file into lines and then split lines into elements
            label = [line.split(" ") for line in label.split("\n")]
            #exclude first colmumn which is the class
            label=[row[1:] for row in label]
            #Convert elements to int beside the last one which is empty
            label = [[int(y) for y in x] for x in label[:-1]]
            #Convert to numpy array
            label=np.array(label)

    # Read the label from ground truth
    label_path_gt="/root/yolov9/data_kitti/kitti_gt/labels/"+filename
    with open(label_path_gt, 'r') as file:
        label_gt = file.read()
        if label_gt == '':
            empty_file = True
            print('Empty file')
        else:
             #Split file into lines and then split lines into elements
            label_gt = [label_gt.split(" ") for label_gt in label_gt.split("\n")] 
            #Exclude first element which is string
            label_gt=[row[1:] for row in label_gt]
            #Convert elements to float beside the last one which is empty
            label_gt = [[float(y) for y in x] for x in label_gt[:-1]]
            #Convert to numpy array
            label_gt=np.array(label_gt)

    #if file is empty skip the iteration
    if empty_file:
        continue

    #If there is detection in the image
    #calculate IoU
    def calculate_iou(box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = box1_area + box2_area - intersection
        iou = intersection / union if union != 0 else 0

        return iou

    #From GT and YOLO labels find the best match
    def find_best_match(box, boxes):
        best_iou = 0
        best_box = None
        counter=0
        number=0
        for candidate_box in boxes:
            iou = calculate_iou(box, candidate_box)
            if iou > best_iou:
                best_iou = iou
                best_box = candidate_box
                number=counter
            counter+=1
        return best_box, best_iou, number

    #Create a list of the best matches
    double=[]
    for i in range(0,label.shape[0]):
        x,y, n=find_best_match(label[i],label_gt[:,:4])

        if x is not None:
            temp=[]
            temp.append(np.array(label_gt[n,4]))
            temp.append(x)
            double.append(temp)

    #define number of detections
    det_number=len(double)

    #for every detection calculate the distance
    for i in range(0,det_number):

        center_x=((double[i][1][2]-double[i][1][0])/2)+double[i][1][0]
        pos=[int(center_x),int(double[i][1][3]),1]

        inv_matrix=np.linalg.inv(matrix)
        res_matrix=inv_matrix@pos

        distance_y=camera_height/res_matrix[1]

        distance_x=abs(res_matrix[0]*distance_y)

        distance_est=np.sqrt(distance_x**2+distance_y**2)

        if distance_est>80:
            continue

        distance_gt.append(float(double[i][0]))
        distance_estimated.append(distance_est)


#plot the results
x = np.linspace(0, 100, 80)
plt.plot(distance_estimated,distance_gt,'o')
plt.plot(x, x, '-c')
# plt.savefig('plot_no_limit.png')
plt.ylabel('Distance GT')
plt.xlabel('Distance Estimated')
plt.show()

# print(len(temp1))



