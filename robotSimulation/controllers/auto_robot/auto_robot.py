"""auto_robot controller."""

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

def yoloLoop():
    frame = imutils.resize(RGBImage, width=400)

    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)

    net.setInput(blob)
    #start = time.time()
    layerOutputs = net.forward(ln)
    #end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
		# loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
                # apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE,THRESHOLD)

	# ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLOURS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                confidences[i])
            cv2.putText(frame, text, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
     # show the output frame
    cv2.imshow("Frame", frame)
     

yolo_directory = "yolo-coco"
# minimum confidence sore
CONFIDENCE = 0.5
# threshold for max supression
THRESHOLD = 0.3
labelsPath = os.path.sep.join([yolo_directory, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLOURS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([yolo_directory, "yolov3.weights"])
configPath = os.path.sep.join([yolo_directory, "yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]



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
  
    left_speed = 3.0
    right_speed = 3.0
    
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
    # plt.imshow(disparity,'gray')
    # plt.show()
    
    yoloLoop()
   

    
    
  
     
     
    cv2.waitKey(timestep)

cv2.destroyAllWindows()
    