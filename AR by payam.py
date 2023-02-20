# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:16:28 2021

@author: ASUS
"""
# used libraries


from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import sys
import numpy as np
import warnings
import cv2.aruco as aruco
import mediapipe as mp
import math


# initiating FPS variables
pTime = 0
cTime = 0

warnings.filterwarnings("ignore")

# input_dictionaries for the number of boxes and their ids
inputs_dict = {"slot 1":[20,21,22,23],
               "slot 2":[8,9,10,11],
               "slot 3":[16,17,18,19],
               "slot 4":[28,29,30,31],
               "slot 5":[3,32,33,34],
               "slot 6":[35,36,37,38],
               "slot 7":[24,25,26,27],
               "slot 8":[12,13,14,15],
               "slot 9":[4,5,6,7],
               "slot 10":[48,49,50,88]
               }

# worker's hand_sign dictionary
hand_sign = [89,90,91,79]

# initiating hand detection detector function
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# distance function for finding the distance between two points
def pyth_dist(y,x,y2,x2):
    dist = math.sqrt( (x-x2)**2 + (y-y2)**2) 
    return dist

# order function for ordering 4 points in a specific distance from (0,0) point
def order_points(four_corners):

    ind = []

    dists = []
    # find tl
    for point in four_corners:
        dists.append(pyth_dist(point[1], point[0], 0, 0))

    ind.append(np.argmin(dists))

    dists = []
    # find tr
    for point in four_corners:
        dists.append(pyth_dist(point[1], point[0], 0, 1000))

    ind.append(np.argmin(dists))

    dists = []
    # find br
    for point in four_corners:
        dists.append(pyth_dist(point[1], point[0], 750, 1000))

    ind.append(np.argmin(dists))

    dists = []
    # find bl
    for point in four_corners:
        dists.append(pyth_dist(point[1], point[0], 750, 0))

    ind.append(np.argmin(dists))
    
    return four_corners[ind]

#  finds the intersection of two lists
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


# detect inices function for matching ids with their corners 
def detect_indices(request, detected):
    intrs_id = []

    for i, p in enumerate(detected):
        if p in request:
            intrs_id.append(i)
            
    return intrs_id

# aruco marker detctor function
def findArucoMarkers(frame,draw = True):
    imgGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    Key = cv2.aruco.DICT_ARUCO_ORIGINAL
    arucoDict = aruco.Dictionary_get(Key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids , rejected = aruco.detectMarkers(imgGray, arucoDict, parameters = arucoParam)
    # print (ids)
    if draw :
        aruco.drawDetectedMarkers(frame,bboxs)

    return [bboxs , ids]

# AR1 fucnction for ARing the entire box
def augmentAruco1 (fourCorners,frame,imgAug, drawId = True):
    
      
    fourCorners = order_points(fourCorners)


    tl = fourCorners[0][0], fourCorners[0][1]
    tr = fourCorners[1][0], fourCorners[1][1]
    br = fourCorners[2][0], fourCorners[2][1]
    bl = fourCorners[3][0], fourCorners[3][1]

    
    if imgAug is not None:
        h, w, c = imgAug.shape
    
    pts1 = np.array([tl,tr,br,bl])  
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    matrix , _ = cv2.findHomography(pts2,pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (frame.shape[1],frame.shape[0]))
    cv2.fillConvexPoly (frame, pts1.astype(int),(0,0,0))
    imgOut = frame + imgOut
    return imgOut
# AR fucnction for ARing the Aruco markers
def augmentAruco (bbox, id, frame,imgAug, drawId = True):
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    
    if imgAug is not None:
        h, w, c = imgAug.shape
    

    pts1 = np.array([tl,tr,br,bl]) 

    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    matrix , _ = cv2.findHomography(pts2,pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (frame.shape[1],frame.shape[0]))
    
    cv2.fillConvexPoly (frame, pts1.astype(int),(0,0,0))
    
    imgOut = frame + imgOut
    return imgOut





#argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
	default="DICT_ARUCO_ORIGINAL",
	help="type of ArUCo tag to detect")
args = vars(ap.parse_args())


# ArUco tag OpenCV dicts
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


# verify that the supplied ArUCo tag 
if ARUCO_DICT.get(args["type"], None) is None:
	print("[INFO] ArUCo tag of '{}' is not supported".format(
		args["type"]))
	sys.exit(0)
# load ArUCo dictionary 
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

# video initialization
vs = VideoStream(src=1).start()
time.sleep(2.0)



# giving inputs

inp = input(str("Enter the name of the slot (ex. slot x) : "))
# comparing the given input with library of boxes 
for i in inputs_dict.keys():
    if inp == i:
        correct_ids = inputs_dict[inp]
    
# Main while loop
        
        imgAug = cv2.imread("13.jpg")
        imgAug2 = cv2.imread("13.jpg")
        while True :
            frame = vs.read()
            frame = imutils.resize(frame, width=1000)
            # print(np.shape(frame))

        	# detect ArUco markers in the input frame
            (corners_f2, ids2, rejected2) = cv2.aruco.detectMarkers(frame,
        		arucoDict, parameters=arucoParams)
        	# verify *at least* one ArUco marker was detected
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            work = 0
            flag = 0
            if len(corners_f2) > 0 :
                ids2 = ids2.flatten()
                ids2 = list(ids2)
                while len(intersection(ids2, hand_sign)) == 1:
                    
                    
                    work= work + 1
                    print(work)
                    cv2.putText(frame," dear worker your id is aproved",(100,500),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
                    cv2.imshow("Frame", frame)
                    cv2.waitKey(1)
                    if work == 1500:
                        flag = 1
                        break
                

            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            if flag == 1:
                break
        while flag == 1:
            

            # taking the frame and resizing it
            frame = vs.read()
            frame = imutils.resize(frame, width=1000)
            # print(np.shape(frame))

        	# detect ArUco markers in the input frame
            (corners_f, ids, rejected) = cv2.aruco.detectMarkers(frame,
        		arucoDict, parameters=arucoParams)
        	# verify *at least* one ArUco marker was detected
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            
            if len(corners_f) > 0:
                ids = ids.flatten()
                ids = list(ids)
                corners_f = np.array(corners_f)
                corners_f = corners_f[detect_indices(correct_ids, ids)]
                ave = np.mean(corners_f,2)
                # if 4 markers detcted
                
                if len(intersection(ids, correct_ids)) == 4 :
                
                    # wrong picking error while detecting correct slot
                    (corners_f1, ids1, rejected1) = cv2.aruco.detectMarkers(frame,arucoDict, parameters=arucoParams)
                    if hand_sign in ids1 and results.multi_hand_landmarks != None :
                        # corners_f1 = np.array(corners_f1)
                        # corners_f1 = corners_f1[detect_indices(hand_sign, ids1)]
                        # fourCorners1 = np.mean(corners_f1, 2)
                        # fourCorners1 = np.int0(fourCorners1)
                        # fourCorners1 = fourCorners1.reshape(4,2)
                        # print(fourCorners1)
                        cv2.putText(frame,"!! Wrong picking slot !!",(300,100 ),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
                    # first method    
                    fourCorners = np.mean(corners_f, 2)
                    fourCorners = np.int0(fourCorners)
                    fourCorners = fourCorners.reshape(4,2)
                    fourCorners = np.array(fourCorners)
                    ind = np.argsort(fourCorners[:,1])
                    fourCorners = fourCorners[ind]

                    frame = augmentAruco1(fourCorners,frame,imgAug2)
                    
                    # second method
                    
                    
                    arucoFound = [corners_f,ids]

                    for bbox , id in zip(arucoFound[0],arucoFound[1]):

                        frame = augmentAruco(bbox,id,frame,imgAug)
                        
                    # third method
                    
                    # cv2.circle(frame, (int(x),int(y)),60, (255, 255, 80), 3)
                    # cv2.circle(frame, (int(x),int(y)), 5, (255, 255, 80), 3)
                     
                    # fourth method
                    
                #     cv2.drawContours(frame, [box], 0, (0,255,0), 3)
                #     cv2.polylines(frame, [box], 0, (0,255,0), 3)
                #     cv2.putText(frame, 'Correct box',
            				# (int(x), int(y - 15)),
            				# cv2.FONT_HERSHEY_SIMPLEX,
            				# 0.5, (0, 255, 0), 2)
                # if 2 markers detcted
                
                elif  len(intersection(ids, correct_ids)) == 2  :
                    
                    arucoFound = [corners_f,ids]
                    for bbox , id in zip(arucoFound[0],arucoFound[1]):

                        frame = augmentAruco(bbox,id,frame,imgAug)
                        
                # if 1 markers detcted
                elif len(intersection(ids, correct_ids)) == 1 :
                    arucoFound = [corners_f,ids]
                    for bbox , id in zip(arucoFound[0],arucoFound[1]):


                        frame = augmentAruco(bbox,id,frame,imgAug)
                # if 3 markers detcted
                elif len(intersection(ids, correct_ids)) == 3 : 
                    arucoFound = [corners_f,ids]
                    for bbox , id in zip(arucoFound[0],arucoFound[1]): 

                        frame = augmentAruco(bbox,id,frame,imgAug)

                if  results.multi_hand_landmarks != None and  len(intersection(ids, correct_ids)) == 0 and  len(intersection(ids, hand_sign)) == 1:
                        cv2.putText(frame,'!! Wrong picking slot !!',(10,500),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
                
            # showing the frame and getting the FPS
            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("e"):
                break

cv2.destroyAllWindows()
vs.stop()            



