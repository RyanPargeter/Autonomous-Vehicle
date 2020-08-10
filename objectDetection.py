
import numpy as np
import argparse
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import time
import imutils
from scipy.spatial import distance
from typing import NamedTuple
import collections
from matplotlib import pyplot as plt
import os
from mpl_toolkits import mplot3d
import math
#%matplotlib inline
from IPython import display


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))





class Object(NamedTuple):
    name: str
    centroidx: float
    centroidy: float
    startX: float
    startY: float
    endX: float
    endY: float
    dissapeared: int
        
    def setDissapeared(self, disp):
        dissapeared = disp




def doObjectDetection(net, frame, real_points, x):

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    (h, w) = frame.shape[:2]
    return getBoundingBoxes(detections, real_points, x, h, w)



def getBoundingBoxes(detections, real_points, state_x, h, w):

   

    new_objects = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.8:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            
            x_centre = ((startX + endX)/2).astype("int")
            y_centre = ((startY + endY)/2).astype("int")  
            
#             if x_centre >= 400:
#                 x_centre = 399
#             if y_centre >= 121:
#                 y_centre = 120
            zarr = []    
            for y in range(startX, endX - 1):
                for x in range(startY, endY - 1):
                    realxyz = real_points[x,y]
                    zarr.append(realxyz[2])
           

            pnts = real_points[y_centre, x_centre]
            print('pnts = ', pnts)
            real_centx , real_centy = (pnts[0]/100) + state_x[1][0], (sum(zarr) / len(zarr))+ state_x[0][0]

            # 
            # 
#             ax.scatter(sum(zarr) / len(zarr), old_objects[obj][1])
#             print(sum(zarr) / len(zarr))
            
    
            
#             if real_centy >= 1:
            new_objects.append(Object(CLASSES[idx], real_centx , real_centy, startX, startY, endX, endY,0))
            
            
    return new_objects




