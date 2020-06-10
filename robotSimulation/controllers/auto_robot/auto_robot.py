"""auto_robot controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Camera
import cv2
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import matplotlib.pyplot as plt

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
cv2.namedWindow("preview")

display = robot.getDisplay("display")
# display.attachCamera(camera0)

count = 0;
while robot.step(timestep) != -1:
  
    left_speed = 3.0
    right_speed = 3.0
    
    left_wheel.setVelocity(left_speed)
    right_wheel.setVelocity(right_speed)
    
    cameraData = camera0.getImage();
    image = np.frombuffer(cameraData, np.uint8).reshape((camera0.getHeight(), camera0.getWidth(), 4))
    img = image.copy()
      
    # cv2.imshow('preview', frame)
    
    # key = cv2.waitKey(timestep) & 0xFF
    
    # if key == ord("q"):
        # break
    img[5][5] = [0,0,0,255]
    print(img[0][0])
    cv2.imshow("preview", img)
    cv2.waitKey(timestep)
# cv2.destroyAllWindows()
    