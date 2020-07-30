"""SLAM controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Camera
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from matplotlib import pyplot as plt
from scipy.spatial import distance    
from sklearn.neighbors import KDTree 

class Landmark:
    id = 0
    x = 0.0
    y = 0.0
    z = 0.0
    
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        
class LandmarkPair:
    id = 0
    landmark_1 = []
    landmark_2 = []
 
    
    def __init__(self, id, landmark_1, landmark_2):
        self.id = id
        self.landmark_1 = landmark_1
        self.landmark_2 = landmark_2


robot = Robot()

timestep = int(robot.getBasicTimeStep())

left_wheel = robot.getMotor('left wheel motor')
right_wheel = robot.getMotor('right wheel motor')

left_wheel.setPosition(float('inf'))
left_wheel.setVelocity(0.0)
right_wheel.setPosition(float('inf'))
right_wheel.setVelocity(0.0)

camera0 = robot.getCamera("camera0");
camera1 = robot.getCamera("camera1");
camera0.enable( 2 * timestep);
camera1.enable( 2 * timestep);


# cv2.startWindowThread()
# cv2.namedWindow("Frame")
#TODO Need a list of official landmarks with number of times observed.
landmarks = {'initial_landmark' : [],
            'next_frame_landmark' : []}
            
landmark_pairs = []

count = 0;
while robot.step(timestep) != -1:
  
    left_speed = 5.0
    right_speed = 5.0
    
    left_wheel.setVelocity(left_speed)
    right_wheel.setVelocity(right_speed)
    
    leftCameraData = camera1.getImage();
    leftImage = np.frombuffer(leftCameraData, np.uint8).reshape((camera1.getHeight(), camera1.getWidth(), 4))
    leftImg = leftImage.copy()
    leftRGBImage = leftImg[:,:,0:3].copy()
    
    cameraData = camera0.getImage();
    image = np.frombuffer(cameraData, np.uint8).reshape((camera0.getHeight(), camera0.getWidth(), 4))
    img = image.copy()
    RGBImage = img[:,:,0:3].copy()
    
    frame0_new = cv2.cvtColor(leftRGBImage, cv2.COLOR_BGR2GRAY) 
    frame1_new = cv2.cvtColor(RGBImage, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(frame1_new,frame0_new)
    
 
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(frame0_new,None).copy() 
    img=cv2.drawKeypoints(frame0_new,kp,leftRGBImage)
   
    pts = [k.pt for k in kp]
    realWorldSift = []
    # if(len(pts) > 0):
        # for point in range(len(pts)):
            # y = int(pts[point][0])
            # x = int(pts[point][1])
            # Z = f*0.04/disparity[x,y]
            # realWorldSift[0].append(x * Z / f)
            # realWorldSift[1].append(y * Z / f)
            # realWorldSift[2].append(Z)
        
    h, w = disparity.shape
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0,  0, w / 2],
                    [0, -1,  0,  h / 2],  # turn points 180 deg around x-axis,
                    [0, 0, f,  0],  # so that y-axis looks up
                    [0, 0,  0,  1]])
    
    real_points = cv2.reprojectImageTo3D(disparity, Q)
    # mask = disparity > disparity.min()
    # real_points = real_points[mask]
    if len(pts) > 0:
        if not(len(landmarks['initial_landmark']) > 0):
            for i in range(len(pts)):
                current_landmark = real_points[int(pts[i][1]),int(pts[i][0])]
                landmarks['initial_landmark'].append(Landmark(i,current_landmark[0], current_landmark[1], current_landmark[2]))
        else:
            num_features = len(landmarks['initial_landmark'])
            landmarks_array = []
            sift_features = []
            for x in range(num_features):
                lmrk = landmarks['initial_landmark'][x]
                landmarks_array.append([lmrk.x, lmrk.y, lmrk.z])
            
            for i in range(len(pts)):
                sift_features.append(real_points[int(pts[i][1]),int(pts[i][0])])
        
            nn_tree = KDTree(np.asarray(sift_features))
           
            nn_index_of_closest = nn_tree.query(np.asarray(landmarks_array), k = 1, return_distance = False)
            for x in range(nn_index_of_closest.shape[0]):
                
                idx = nn_index_of_closest[x, 0]
             

                landmark_pairs.append(LandmarkPair(idx,sift_features[idx],landmarks_array[x]))  


            plt.scatter(sift_features)
            plt.show()
                
  
