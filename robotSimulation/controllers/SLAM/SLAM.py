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

cv2.startWindowThread()
cv2.namedWindow("Frame")


count = 0;
while robot.step(timestep) != -1:
  
    left_speed = 5.0
    right_speed = 5.0
    
    left_wheel.setVelocity(left_speed)
    right_wheel.setVelocity(right_speed)
    
    leftCameraData = camera1.getImage();
    leftImage = np.frombuffer(leftCameraData, np.uint8).reshape((camera1.getHeight(), camera1.getWidth(), 4))
    leftImg = leftImage.copy()
    leftRGBImage = leftImg[:,:,0:3]
    
    cameraData = camera0.getImage();
    image = np.frombuffer(cameraData, np.uint8).reshape((camera0.getHeight(), camera0.getWidth(), 4))
    img = image.copy()
    RGBImage = img[:,:,0:3]
    
    frame0_new = cv2.cvtColor(leftRGBImage, cv2.COLOR_BGR2GRAY)
    frame1_new = cv2.cvtColor(RGBImage, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(frame0_new,frame1_new)

    
    cv2.imshow('Frame', np.asarray(stereo))
    cv2.waitKey(timestep)

cv2.destroyAllWindows()
    
