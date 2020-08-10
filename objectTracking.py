
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



def trackObjects(objectId,old_objects,new_objects, state_x):
    if len(old_objects) == 0 :
        
        objectId,old_objects,new_objects = trackAllObjects(objectId,old_objects,new_objects, state_x)
        
    else:
        unusedRows = set(range(0, len(old_objects)))
        lst_objectIDs = list(old_objects.keys())
        
        if len(new_objects) > 0:
                
            old_centroids, new_centroids = extractCentroids(old_objects, new_objects, state_x)

            dist = distance.cdist(np.asarray(old_centroids), np.asarray(new_centroids))

            old_idx = dist.min(axis=1).argsort()
            new_idx = dist.argmin(axis=1)[old_idx]
            
            
           

            old_objects, unusedRows, unusedCols, new_objects, objectId = updateExistingObjectLocations(old_idx, new_idx, lst_objectIDs, old_objects, new_objects, dist, objectId, state_x)
                
#             else:
#                 old_objects, new_objects, objectId = trackNewObjects(unusedCols, old_objects, new_objects, objectId)
            
        if len(old_objects) > len(new_objects):
# dist.shape[0] >= dist.shape[1]
            old_objects = dealWithDissapearedObjects(unusedRows, lst_objectIDs, old_objects)

#             else:
                
#                 old_objects, new_objects, objectId = trackNewObjects(unusedCols, old_objects, new_objects, objectId)  
                    
    return objectId,old_objects,new_objects








def trackNewObjects(unusedCols, old_objects, new_objects, objectId, state_x):
    
    for obj in sorted(range(len(new_objects)), reverse=True):
        
        if new_objects[obj][2] - state_x[0][0]  > 1:
            #
            old_objects[objectId] = new_objects[obj]
            objectId += 1
        else:
            del new_objects[obj]
            
    return old_objects, new_objects, objectId


def dealWithDissapearedObjects(unusedRows, lst_objectIDs, old_objects):
    
    for row in unusedRows:
        
        obj_id = lst_objectIDs[row]
        dissapearance_val = old_objects[obj_id][7]
        old_objects[obj_id] = old_objects[obj_id]._replace(dissapeared=dissapearance_val + 1)


        if old_objects[obj_id][7] > 50:
            del old_objects[obj_id]
            
    return old_objects






def updateExistingObjectLocations(old_idx, new_idx, lst_objectIDs, old_objects, new_objects, dist, objectId, state_x):
    usedRows = set()
    usedCols = set()
    deletedCols = set()
    
    for (row, col) in sorted(zip(old_idx, new_idx),reverse=True):
        if row not in usedRows and col not in usedCols:

            if new_objects[col][2]  - state_x[0][0] > 1:
                #
                #print('distance = ',dist[row,col])
                if dist[row,col] < 5:
                    obj_id = lst_objectIDs[row]
                    old_objects[obj_id] = new_objects[col]
                    old_objects[obj_id] = old_objects[obj_id]._replace(dissapeared=0)
                    usedRows.add(row)
                    usedCols.add(col)
            else:
                del new_objects[col]
                usedCols.add(col)
                deletedCols.add(col)
#     print('dst shape 0 = ', dist.shape[0])
#     print('used rows = ', usedRows)
    unusedRows = set(range(0, dist.shape[0])).difference(usedRows)
    unusedCols = set(range(0, dist.shape[1])).difference(usedCols)
   # print('unused rows = ',unusedRows)
    
    for index in sorted(usedCols, reverse=True):
        if len(new_objects) > 0 and index not in deletedCols:
            del new_objects[index]
            
    old_objects, new_objects, objectId = trackNewObjects(unusedCols, old_objects, new_objects, objectId, state_x)
    
    return old_objects, unusedRows, unusedCols, new_objects, objectId






def extractCentroids(old_objects, new_objects, state_x):
    old_centroids = []
    for x in old_objects:
        old_centroids.append([old_objects[x][1],old_objects[x][2]])

    new_centroids = []
    for x in new_objects:
        new_centroids.append([x[1]   ,x[2]])
        #+ state_x[1][0]
        # + state_x[0][0]
    return old_centroids, new_centroids






def trackAllObjects(objectId,old_objects,new_objects, state_x):
    for x in range(len(new_objects)):
        if new_objects[x][2] - state_x[0][0] > 1:
            #
            old_objects[objectId] = new_objects[x]
            objectId += 1
        else:
            del new_objects[x]
    return objectId,old_objects,new_objects










